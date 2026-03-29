// Test TMD mask with INVERTED semantics (bit set = enabled)
// Also tests writing to the 0x000fffff field directly
//
// Usage: ./tmd_test_inv <dev> <offset> <mask_lo> <mask_hi> [enable_off] [enable_bit]

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

static int g_active = 0;
static int g_mask_off = -1;
static uint32_t g_mask_lo = 0;
static uint32_t g_mask_hi = 0;
static int g_enable_off = -1;
static uint32_t g_enable_bit = 0x80000000;

__global__ void read_smid(uint8_t* out) {
    if (threadIdx.x != 0) return;
    int smid;
    asm("mov.u32 %0, %%smid;" : "=r"(smid));
    out[blockIdx.x] = smid;
}

static void cb(void *u, int domain, int cbid, const void *in_params) {
    if (!g_active) return;
    void* tmd = *((void**)in_params + 4);
    if (!tmd) return;

    if (g_mask_off >= 0) {
        *(uint32_t*)((uint8_t*)tmd + g_mask_off) = g_mask_lo;
        *(uint32_t*)((uint8_t*)tmd + g_mask_off + 4) = g_mask_hi;
    }
    if (g_enable_off >= 0) {
        *(uint32_t*)((uint8_t*)tmd + g_enable_off) |= g_enable_bit;
    }
}

int main(int argc, char** argv) {
    if (argc < 5) {
        printf("Usage: %s <dev> <mask_offset> <mask_lo> <mask_hi> [enable_off] [enable_bit]\n", argv[0]);
        printf("       %s <dev> --baseline\n", argv[0]);
        return 2;
    }

    int dev = atoi(argv[1]);
    int baseline = !strcmp(argv[2], "--baseline");

    cudaSetDevice(dev);

    uintptr_t* tbl;
    cuGetExportTable((const void**)&tbl, &callback_funcs_id);
    auto sub = (int(*)(uint32_t*, void(*)(void*,int,int,const void*), void*))*(tbl+3);
    auto en  = (int(*)(uint32_t,uint32_t,int,int))*(tbl+6);
    uint32_t h;
    sub(&h, cb, NULL);
    en(1, h, QMD_DOMAIN, QMD_PRE_UPLOAD);

    if (!baseline) {
        g_mask_off = strtol(argv[2], NULL, 0);
        g_mask_lo = strtoul(argv[3], NULL, 0);
        g_mask_hi = strtoul(argv[4], NULL, 0);
        if (argc > 5) g_enable_off = strtol(argv[5], NULL, 0);
        if (argc > 6) g_enable_bit = strtoul(argv[6], NULL, 0);
        g_active = 1;
    }

    uint8_t *d_sm, h_sm[NUM_BLOCKS];
    cudaMalloc(&d_sm, NUM_BLOCKS);
    cudaStream_t s;
    cudaStreamCreate(&s);

    read_smid<<<NUM_BLOCKS, 1024, 0, s>>>(d_sm);
    cudaError_t err = cudaStreamSynchronize(s);
    if (err != cudaSuccess) {
        printf("-1\n");
        return 1;
    }

    cudaMemcpy(h_sm, d_sm, NUM_BLOCKS, cudaMemcpyDeviceToHost);

    // Sort
    for (int i = 0; i < NUM_BLOCKS; i++)
        for (int j = i+1; j < NUM_BLOCKS; j++)
            if (h_sm[i] > h_sm[j]) { uint8_t t = h_sm[i]; h_sm[i] = h_sm[j]; h_sm[j] = t; }

    int uniq = 1;
    for (int i = 1; i < NUM_BLOCKS; i++)
        if (h_sm[i] != h_sm[i-1]) uniq++;

    printf("%d", uniq);
    if (uniq <= 10) {
        printf(" sms:");
        uint8_t prev = 255;
        for (int i = 0; i < NUM_BLOCKS; i++)
            if (h_sm[i] != prev) { printf(" %d", h_sm[i]); prev = h_sm[i]; }
    }
    printf("\n");

    return 0;
}
