// Verify that TMD writes in callback actually persist.
// Uses two kernel launches: first to modify TMD, second to read it back.
// Also explores different callback domains and cbids.

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

static int g_call_count = 0;
static int g_phase = 0;  // 0=dump pre, 1=modify, 2=dump post

__global__ void read_smid(uint8_t* out) {
    if (threadIdx.x != 0) return;
    int smid;
    asm("mov.u32 %0, %%smid;" : "=r"(smid));
    out[blockIdx.x] = smid;
}

static void verify_callback(void *u, int domain, int cbid, const void *in_params) {
    g_call_count++;
    uint32_t sz = *(uint32_t*)in_params;
    void* tmd = *((void**)in_params + 4);
    if (!tmd) { printf("  [call %d] TMD is NULL!\n", g_call_count); return; }

    uint8_t* t = (uint8_t*)tmd;

    if (g_phase == 0) {
        // Dump key areas before modification
        printf("  [call %d] PRE-MODIFY domain=%d cbid=%d tmd=%p\n", g_call_count, domain, cbid);
        printf("    TMD+0x000: %08x %08x %08x %08x\n",
               *(uint32_t*)(t), *(uint32_t*)(t+4), *(uint32_t*)(t+8), *(uint32_t*)(t+12));
        printf("    TMD+0x048: %08x (version field)\n", *(uint32_t*)(t+0x48));
        printf("    TMD+0x130: %08x %08x (Hopper mask area)\n",
               *(uint32_t*)(t+0x130), *(uint32_t*)(t+0x134));
        printf("    TMD+0x278: %08x %08x (candidate mask)\n",
               *(uint32_t*)(t+0x278), *(uint32_t*)(t+0x27c));

        // Write marker at offset 0x278
        *(uint32_t*)(t + 0x278) = 0xDEADBEEF;
        *(uint32_t*)(t + 0x130) = 0xCAFEBABE;

        printf("    WROTE 0xDEADBEEF at TMD+0x278\n");
        printf("    WROTE 0xCAFEBABE at TMD+0x130\n");

        // Read back immediately
        printf("    READBACK TMD+0x278: %08x\n", *(uint32_t*)(t+0x278));
        printf("    READBACK TMD+0x130: %08x\n", *(uint32_t*)(t+0x130));
    }
    else if (g_phase == 1) {
        // Check if TMD address is the same (reused) or different
        printf("  [call %d] SECOND LAUNCH domain=%d cbid=%d tmd=%p\n", g_call_count, domain, cbid);
        printf("    TMD+0x278: %08x (was 0xDEADBEEF if same buffer)\n", *(uint32_t*)(t+0x278));
        printf("    TMD+0x130: %08x (was 0xCAFEBABE if same buffer)\n", *(uint32_t*)(t+0x130));
    }
}

// Try all combinations of domain/cbid to find if there are other callbacks
static void enumerate_callback(void *u, int domain, int cbid, const void *in_params) {
    printf("  Callback: domain=0x%x cbid=0x%x param_size=0x%x\n",
           domain, cbid, *(uint32_t*)in_params);
}

int main(int argc, char** argv) {
    int dev = 0;
    if (argc > 1) dev = atoi(argv[1]);

    cudaSetDevice(dev);

    int major, minor, num_sms;
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
    cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev);
    cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, dev);
    printf("GPU: sm_%d%d, %d SMs\n\n", major, minor, num_sms);

    uintptr_t* tbl;
    cuGetExportTable((const void**)&tbl, &callback_funcs_id);
    auto sub = (int(*)(uint32_t*, void(*)(void*,int,int,const void*), void*))*(tbl+3);
    auto en  = (int(*)(uint32_t,uint32_t,int,int))*(tbl+6);

    // Part 1: Verify TMD write persistence
    printf("=== Part 1: TMD Write Verification ===\n");
    uint32_t h;
    sub(&h, verify_callback, NULL);
    en(1, h, QMD_DOMAIN, QMD_PRE_UPLOAD);

    uint8_t *d_sm;
    cudaMalloc(&d_sm, NUM_BLOCKS);
    cudaStream_t s;
    cudaStreamCreate(&s);

    printf("First launch:\n");
    g_phase = 0;
    read_smid<<<NUM_BLOCKS, 1024, 0, s>>>(d_sm);
    cudaStreamSynchronize(s);

    printf("\nSecond launch:\n");
    g_phase = 1;
    read_smid<<<NUM_BLOCKS, 1024, 0, s>>>(d_sm);
    cudaStreamSynchronize(s);

    // Part 2: Try other domain/cbid combinations
    printf("\n=== Part 2: Enumerate Other Callback Types ===\n");

    // Disable current callback
    en(0, h, QMD_DOMAIN, QMD_PRE_UPLOAD);

    uint32_t h2;
    sub(&h2, enumerate_callback, NULL);

    // Try different domains and cbids
    int domains[] = {0x1, 0x2, 0x3, 0x4, 0x5, 0x6, 0x7, 0x8, 0x9, 0xa, 0xb, 0xc, 0xd, 0xe, 0xf, 0x10};
    int cbids[] = {0x1, 0x2, 0x3, 0x4, 0x5};

    for (int di = 0; di < 16; di++) {
        for (int ci = 0; ci < 5; ci++) {
            int res = en(1, h2, domains[di], cbids[ci]);
            if (res == 0) {
                printf("  Enabled: domain=0x%x cbid=0x%x\n", domains[di], cbids[ci]);
            }
        }
    }

    printf("\nLaunching kernel to see which callbacks fire:\n");
    read_smid<<<NUM_BLOCKS, 1024, 0, s>>>(d_sm);
    cudaStreamSynchronize(s);

    cudaFree(d_sm);
    cudaStreamDestroy(s);

    printf("\n=== Done ===\n");
    return 0;
}
