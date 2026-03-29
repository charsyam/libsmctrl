// Dump all in_params pointer targets and test writing mask to each one
// Also test if pointer at index 3 (not 4) is the actual writable TMD

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

static int g_phase = 0;

__global__ void read_smid(uint8_t* out) {
    if (threadIdx.x != 0) return;
    int smid;
    asm("mov.u32 %0, %%smid;" : "=r"(smid));
    out[blockIdx.x] = smid;
}

static void dump_words(const char* label, void* ptr, int offset, int count) {
    uint32_t* w = (uint32_t*)((uint8_t*)ptr + offset);
    printf("  %s+0x%03x:", label, offset);
    for (int i = 0; i < count; i++) printf(" %08x", w[i]);
    printf("\n");
}

static void my_cb(void *u, int domain, int cbid, const void *in_params) {
    uint32_t sz = *(uint32_t*)in_params;
    int nptrs = sz / sizeof(void*);

    if (g_phase == 0) {
        printf("=== Callback: domain=%d cbid=%d nptrs=%d ===\n", domain, cbid, nptrs);

        // Dump each pointer's data
        for (int idx = 1; idx <= 4 && idx < nptrs; idx++) {
            void* ptr = *((void**)in_params + idx);
            uintptr_t addr = (uintptr_t)ptr;
            if (addr < 0x10000) continue;

            printf("\n--- in_params[%d] = %p ---\n", idx, ptr);
            // Dump first 384 bytes as 32-bit words
            uint32_t* words = (uint32_t*)ptr;
            for (int row = 0; row < 384; row += 32) {
                printf("  0x%03x:", row);
                for (int c = 0; c < 8; c++) {
                    printf(" %08x", words[row/4 + c]);
                }
                printf("\n");
            }
        }

        // Also try dereferencing pointer at index 4 as pointer-to-pointer
        void* ptr4 = *((void**)in_params + 4);
        if (ptr4) {
            // Check if ptr4 itself contains a pointer (double indirection)
            void* ptr4_deref = *(void**)ptr4;
            uintptr_t deref_addr = (uintptr_t)ptr4_deref;
            printf("\n--- Dereferencing ptr[4] -> %p ---\n", ptr4_deref);
            if (deref_addr > 0x10000 && deref_addr < 0xFFFFFFFFFFFFULL) {
                printf("  Looks like a valid pointer! Dumping:\n");
                uint32_t* dw = (uint32_t*)ptr4_deref;
                for (int row = 0; row < 256; row += 32) {
                    printf("  0x%03x:", row);
                    for (int c = 0; c < 8; c++) printf(" %08x", dw[row/4+c]);
                    printf("\n");
                }
            }
        }
    }
}

int main(int argc, char** argv) {
    int dev = 0;
    if (argc > 1) dev = atoi(argv[1]);

    cudaSetDevice(dev);
    printf("CUDA 13.0 TMD structure dump - all pointers\n\n");

    uintptr_t* tbl;
    cuGetExportTable((const void**)&tbl, &callback_funcs_id);
    auto sub = (int(*)(uint32_t*, void(*)(void*,int,int,const void*), void*))*(tbl+3);
    auto en  = (int(*)(uint32_t,uint32_t,int,int))*(tbl+6);
    uint32_t h;
    sub(&h, my_cb, NULL);
    en(1, h, QMD_DOMAIN, QMD_PRE_UPLOAD);

    uint8_t *d_sm;
    cudaMalloc(&d_sm, NUM_BLOCKS);
    cudaStream_t s;
    cudaStreamCreate(&s);

    g_phase = 0;
    read_smid<<<NUM_BLOCKS, 1024, 0, s>>>(d_sm);
    cudaStreamSynchronize(s);

    cudaFree(d_sm);
    cudaStreamDestroy(s);
    return 0;
}
