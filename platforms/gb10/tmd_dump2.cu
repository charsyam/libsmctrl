// Extended TMD dump - dumps ALL in_params pointers and tries
// different interpretations of the TMD structure.
// Also tests inverted mask semantics (bit set = enabled vs disabled).

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

static const CUuuid callback_funcs_id = {
    0x2c, (char)0x8e, 0x0a, (char)0xd8,
    0x07, 0x10, (char)0xab, 0x4e,
    (char)0x90, (char)0xdd, 0x54, 0x71,
    (char)0x9f, (char)0xe5, (char)0xf7, 0x4b
};

#define QMD_DOMAIN 0xb
#define QMD_PRE_UPLOAD 0x1
#define NUM_BLOCKS 142

static int g_mode = 0;      // 0=dump, 1=test_ptr_idx, 2=test_enable_scan
static int g_ptr_idx = 4;   // which in_params pointer to use as TMD
static int g_mask_off = 0x130;
static int g_enable_off = 0;
static int g_inverted = 0;  // 1=inverted mask semantics (bit set = enabled)
static int g_dump_count = 0;

__global__ void read_smid(uint8_t* out) {
    if (threadIdx.x != 0) return;
    int smid;
    asm("mov.u32 %0, %%smid;" : "=r"(smid));
    out[blockIdx.x] = smid;
}

static void dump_mem(const char* label, void* ptr, int bytes) {
    uint8_t* p = (uint8_t*)ptr;
    printf("\n=== %s (ptr=%p, %d bytes) ===\n", label, ptr, bytes);
    for (int row = 0; row < bytes; row += 16) {
        printf("0x%04x: ", row);
        for (int c = 0; c < 16 && row+c < bytes; c++) {
            printf("%02x ", p[row+c]);
            if (c == 7) printf(" ");
        }
        printf("\n");
    }
}

static void my_callback(void *u, int domain, int cbid, const void *in_params) {
    if (g_mode == 0 && g_dump_count == 0) {
        g_dump_count++;
        uint32_t sz = *(uint32_t*)in_params;
        int nptrs = sz / sizeof(void*);

        printf("in_params size: 0x%x (%d bytes, %d ptrs)\n", sz, sz, nptrs);

        // Dump data from each valid-looking pointer
        for (int i = 0; i < nptrs && i < 16; i++) {
            void* ptr = *((void**)in_params + i);
            uintptr_t addr = (uintptr_t)ptr;

            // Skip nulls and small values
            if (addr < 0x10000) {
                printf("[%2d] = 0x%lx (skip - small value)\n", i, addr);
                continue;
            }

            // Try to read first 64 bytes
            printf("[%2d] = %p", i, ptr);

            // Dump first 128 bytes of each pointer
            uint8_t buf[128];
            // Note: this might segfault for stack pointers, but let's try
            memcpy(buf, ptr, 128);
            printf(" first 32 words:");
            uint32_t* words = (uint32_t*)buf;
            printf("\n");
            for (int w = 0; w < 32; w++) {
                if (w % 8 == 0) printf("       +0x%03x: ", w*4);
                printf("%08x ", words[w]);
                if (w % 8 == 7) printf("\n");
            }
        }
        return;
    }

    if (g_mode == 1 || g_mode == 2) {
        uint32_t sz = *(uint32_t*)in_params;
        int nptrs = sz / sizeof(void*);
        if (g_ptr_idx >= nptrs) return;

        void* base = *((void**)in_params + g_ptr_idx);
        if (!base) return;

        uint32_t mask_lo, mask_hi;
        if (g_inverted) {
            mask_lo = 0x00000001;  // enable TPC 0 only
            mask_hi = 0x00000000;
        } else {
            mask_lo = 0xFFFFFFFE;  // disable all except TPC 0
            mask_hi = 0xFFFFFFFF;
        }

        *(uint32_t*)((uint8_t*)base + g_mask_off) = mask_lo;
        *(uint32_t*)((uint8_t*)base + g_mask_off + 4) = mask_hi;

        if (g_mode == 2 && g_enable_off >= 0) {
            *(uint32_t*)((uint8_t*)base + g_enable_off) |= 0x80000000;
        }
    }
}

static int test_run(int dev) {
    uint8_t *d_sm, h_sm[NUM_BLOCKS];
    cudaMalloc(&d_sm, NUM_BLOCKS);
    cudaStream_t s;
    cudaStreamCreate(&s);

    read_smid<<<NUM_BLOCKS, 1024, 0, s>>>(d_sm);
    cudaError_t err = cudaStreamSynchronize(s);
    if (err != cudaSuccess) {
        cudaGetLastError();
        cudaFree(d_sm);
        cudaStreamDestroy(s);
        return -1;
    }

    cudaMemcpy(h_sm, d_sm, NUM_BLOCKS, cudaMemcpyDeviceToHost);

    // Sort and count unique
    for (int i = 0; i < NUM_BLOCKS; i++)
        for (int j = i+1; j < NUM_BLOCKS; j++)
            if (h_sm[i] > h_sm[j]) { uint8_t t = h_sm[i]; h_sm[i] = h_sm[j]; h_sm[j] = t; }

    int uniq = 1;
    for (int i = 1; i < NUM_BLOCKS; i++)
        if (h_sm[i] != h_sm[i-1]) uniq++;

    cudaFree(d_sm);
    cudaStreamDestroy(s);
    return uniq;
}

int main(int argc, char** argv) {
    int dev = 0;
    if (argc > 1) dev = atoi(argv[1]);

    cudaSetDevice(dev);

    int major, minor, num_sms;
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
    cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev);
    cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, dev);
    int ver;
    cuDriverGetVersion(&ver);
    printf("GPU: sm_%d%d, %d SMs, CUDA %d.%d\n\n", major, minor, num_sms,
           ver / 1000, (ver % 1000) / 10);

    // Register callback
    uintptr_t* tbl;
    cuGetExportTable((const void**)&tbl, &callback_funcs_id);
    auto sub = (int(*)(uint32_t*, void(*)(void*,int,int,const void*), void*))*(tbl+3);
    auto en  = (int(*)(uint32_t,uint32_t,int,int))*(tbl+6);
    uint32_t h;
    sub(&h, my_callback, NULL);
    en(1, h, QMD_DOMAIN, QMD_PRE_UPLOAD);

    // Phase 1: Dump all pointers
    printf("=== Phase 1: Dump all in_params pointers ===\n");
    g_mode = 0;
    test_run(dev);

    // Phase 2: Test each pointer index as potential TMD
    printf("\n=== Phase 2: Test each pointer as TMD ===\n");
    printf("(Writing mask at offsets 0x130 and 0x278, with enable at TMD+0)\n");
    for (int ptr_idx = 1; ptr_idx <= 4; ptr_idx++) {
        for (int mask_off : {0x054, 0x084, 0x130, 0x278}) {
            for (int inv = 0; inv <= 1; inv++) {
                g_mode = 2;
                g_ptr_idx = ptr_idx;
                g_mask_off = mask_off;
                g_enable_off = 0;
                g_inverted = inv;

                int r = test_run(dev);
                if (r > 0 && r <= 4) {
                    printf("  ** HIT ** ptr[%d] mask_off=0x%x inv=%d: %d SMs\n",
                           ptr_idx, mask_off, inv, r);
                }
            }
        }
    }

    // Phase 3: Scan enable bit location with mask at various offsets
    printf("\n=== Phase 3: Scan enable bit locations ===\n");
    printf("(ptr=4, testing enable at various offsets)\n");
    g_ptr_idx = 4;
    for (int mask_off : {0x084, 0x130, 0x278}) {
        for (int enable_off = 0; enable_off < 0x100; enable_off += 4) {
            for (int inv = 0; inv <= 1; inv++) {
                g_mode = 2;
                g_mask_off = mask_off;
                g_enable_off = enable_off;
                g_inverted = inv;

                int r = test_run(dev);
                if (r > 0 && r <= 4) {
                    printf("  ** HIT ** enable=0x%x mask=0x%x inv=%d: %d SMs\n",
                           enable_off, mask_off, inv, r);
                }
            }
        }
    }

    // Phase 4: Combined scan - both enable and mask offsets
    printf("\n=== Phase 4: Combined enable+mask scan (focused) ===\n");
    g_ptr_idx = 4;
    for (int enable_off = 0; enable_off < 0x60; enable_off += 4) {
        for (int mask_off = 0x050; mask_off < 0x360; mask_off += 4) {
            g_mode = 2;
            g_mask_off = mask_off;
            g_enable_off = enable_off;
            g_inverted = 0;

            int r = test_run(dev);
            if (r > 0 && r <= 4) {
                printf("  ** HIT ** enable=0x%x mask=0x%x: %d SMs\n",
                       enable_off, mask_off, r);
            }
        }
    }

    printf("\n=== All scans complete ===\n");
    return 0;
}
