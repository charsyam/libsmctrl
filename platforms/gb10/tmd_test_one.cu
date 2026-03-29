// Test a single TMD offset - returns exit code 0 if mask works
// Usage: ./tmd_test_one <device> <offset> <mode>
//   mode: 1=mask, 2=mask+enable, 3=mask+enable+ext

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

static int g_offset = -1;
static int g_mode = 0;

__global__ void read_smid(uint8_t* out) {
    if (threadIdx.x != 0) return;
    int smid;
    asm("mov.u32 %0, %%smid;" : "=r"(smid));
    out[blockIdx.x] = smid;
}

static void cb(void *u, int domain, int cbid, const void *in_params) {
    if (g_mode == 0) return;
    void* tmd = *((void**)in_params + 4);
    if (!tmd) return;

    int off = g_offset;
    uint32_t lower = 0xFFFFFFFE;  // enable only TPC 0
    uint32_t upper = 0xFFFFFFFF;

    *(uint32_t*)((uint8_t*)tmd + off) = lower;
    *(uint32_t*)((uint8_t*)tmd + off + 4) = upper;

    if (g_mode >= 2)
        *(uint32_t*)tmd |= 0x80000000;

    if (g_mode >= 3) {
        *(uint32_t*)((uint8_t*)tmd + off + 8) = 0xFFFFFFFF;
        *(uint32_t*)((uint8_t*)tmd + off + 12) = 0xFFFFFFFF;
    }
}

int main(int argc, char** argv) {
    if (argc < 4) {
        printf("Usage: %s <dev> <offset> <mode> [--baseline]\n", argv[0]);
        return 2;
    }

    int dev = atoi(argv[1]);
    int offset = strtol(argv[2], NULL, 0);
    int mode = atoi(argv[3]);
    int baseline_only = (argc > 4 && !strcmp(argv[4], "--baseline"));

    cudaSetDevice(dev);

    uintptr_t* tbl;
    cuGetExportTable((const void**)&tbl, &callback_funcs_id);
    auto sub = (int(*)(uint32_t*, void(*)(void*,int,int,const void*), void*))*(tbl+3);
    auto en  = (int(*)(uint32_t,uint32_t,int,int))*(tbl+6);
    uint32_t h;
    sub(&h, cb, NULL);
    en(1, h, QMD_DOMAIN, QMD_PRE_UPLOAD);

    uint8_t *d_sm, h_sm[NUM_BLOCKS];
    cudaMalloc(&d_sm, NUM_BLOCKS);
    cudaStream_t s;
    cudaStreamCreate(&s);

    if (baseline_only) {
        g_mode = 0;
    } else {
        g_offset = offset;
        g_mode = mode;
    }

    read_smid<<<NUM_BLOCKS, 1024, 0, s>>>(d_sm);
    cudaError_t err = cudaStreamSynchronize(s);
    if (err != cudaSuccess) {
        printf("-1\n");
        return 1;
    }

    cudaMemcpy(h_sm, d_sm, NUM_BLOCKS, cudaMemcpyDeviceToHost);

    // Count unique & list SMs
    uint8_t sorted[NUM_BLOCKS];
    memcpy(sorted, h_sm, NUM_BLOCKS);
    for (int i = 0; i < NUM_BLOCKS; i++)
        for (int j = i+1; j < NUM_BLOCKS; j++)
            if (sorted[i] > sorted[j]) { uint8_t t = sorted[i]; sorted[i] = sorted[j]; sorted[j] = t; }

    int uniq = 1;
    for (int i = 1; i < NUM_BLOCKS; i++)
        if (sorted[i] != sorted[i-1]) uniq++;

    printf("%d", uniq);
    if (uniq <= 8) {
        printf(" sms:");
        uint8_t prev = 255;
        for (int i = 0; i < NUM_BLOCKS; i++)
            if (sorted[i] != prev) { printf(" %d", sorted[i]); prev = sorted[i]; }
    }
    printf("\n");

    return 0;
}
