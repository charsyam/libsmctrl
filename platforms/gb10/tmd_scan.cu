// TMD offset scanner for Blackwell / CUDA 13.0
// Brute-force scans TMD offsets to find the TPC mask field location.
//
// Strategy:
// 1. For each candidate offset, hook the TMD callback
// 2. Write a restrictive TPC mask at that offset
// 3. Check if the kernel was actually constrained to fewer SMs
//
// Build:
//   nvcc -ccbin g++ tmd_scan.cu -o tmd_scan -lcuda -I/usr/local/cuda/include -L/usr/local/cuda/lib64
// Usage:
//   ./tmd_scan [device_id]

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
#define MAX_SMS 256

// Scan parameters
static int g_scan_offset = -1;        // Current offset being tested
static int g_scan_mode = 0;           // 0=disabled, 1=write mask, 2=write mask+enable
static uint32_t g_enable_offset = 0;  // Offset for enable bit
static int g_tmd_size = 0;            // TMD size estimate

__global__ void read_smid_kernel(uint8_t* smid_arr) {
    if (threadIdx.x != 0) return;
    int smid;
    asm("mov.u32 %0, %%smid;" : "=r"(smid));
    smid_arr[blockIdx.x] = smid;
}

static void scan_callback(void *ukwn, int domain, int cbid, const void *in_params) {
    if (g_scan_mode == 0) return;

    void* tmd = *((void**)in_params + 4);
    if (!tmd) return;

    if (g_scan_offset >= 0 && g_scan_offset < 900) {
        // Write a mask that enables only TPC 0 (disable all others)
        // TPC mask: bit set = disabled, so ~0x1 = disable all except TPC 0
        uint32_t lower_mask = 0xFFFFFFFE;  // disable all except TPC 0
        uint32_t upper_mask = 0xFFFFFFFF;  // disable all upper TPCs

        *(uint32_t*)((uint8_t*)tmd + g_scan_offset) = lower_mask;
        *(uint32_t*)((uint8_t*)tmd + g_scan_offset + 4) = upper_mask;

        if (g_scan_mode == 2) {
            // Also try setting enable bit at TMD+0
            *(uint32_t*)tmd |= 0x80000000;
        }
        if (g_scan_mode == 3) {
            // Try enable bit + extended mask (4 words)
            *(uint32_t*)((uint8_t*)tmd + g_scan_offset) = lower_mask;
            *(uint32_t*)((uint8_t*)tmd + g_scan_offset + 4) = upper_mask;
            *(uint32_t*)((uint8_t*)tmd + g_scan_offset + 8) = 0xFFFFFFFF;
            *(uint32_t*)((uint8_t*)tmd + g_scan_offset + 12) = 0xFFFFFFFF;
            *(uint32_t*)tmd |= 0x80000000;
        }
    }
}

static int sort_asc(const void* a, const void* b) {
    return *(uint8_t*)a - *(uint8_t*)b;
}

static int count_unique_sms(uint8_t* arr, int len) {
    uint8_t tmp[NUM_BLOCKS];
    memcpy(tmp, arr, len);
    qsort(tmp, len, 1, sort_asc);
    int n = 1;
    for (int i = 1; i < len; i++) {
        if (tmp[i] != tmp[i - 1]) n++;
    }
    return n;
}

static int get_max_sm(uint8_t* arr, int len) {
    uint8_t max_sm = 0;
    for (int i = 0; i < len; i++) {
        if (arr[i] > max_sm) max_sm = arr[i];
    }
    return max_sm;
}

int main(int argc, char** argv) {
    int dev = 0;
    if (argc > 1) dev = atoi(argv[1]);

    cudaSetDevice(dev);

    int major, minor, num_sms;
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
    cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev);
    cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, dev);
    int cuda_ver;
    cuDriverGetVersion(&cuda_ver);
    printf("GPU: sm_%d%d, %d SMs, CUDA %d.%d\n", major, minor, num_sms,
           cuda_ver / 1000, (cuda_ver % 1000) / 10);

    // Set up callback
    uintptr_t* tbl_base;
    cuGetExportTable((const void**)&tbl_base, &callback_funcs_id);
    auto subscribe = (int (*)(uint32_t*, void(*)(void*, int, int, const void*), void*))*(tbl_base + 3);
    auto enable_cb = (int (*)(uint32_t, uint32_t, int, int))*(tbl_base + 6);

    uint32_t hndl;
    subscribe(&hndl, scan_callback, NULL);
    enable_cb(1, hndl, QMD_DOMAIN, QMD_PRE_UPLOAD);

    uint8_t *d_smids, *h_smids;
    cudaMalloc(&d_smids, NUM_BLOCKS);
    h_smids = (uint8_t*)malloc(NUM_BLOCKS);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // First: baseline (no modification)
    g_scan_mode = 0;
    read_smid_kernel<<<NUM_BLOCKS, 1024, 0, stream>>>(d_smids);
    cudaStreamSynchronize(stream);
    cudaMemcpy(h_smids, d_smids, NUM_BLOCKS, cudaMemcpyDeviceToHost);
    int baseline_unique = count_unique_sms(h_smids, NUM_BLOCKS);
    printf("Baseline: %d unique SMs\n\n", baseline_unique);

    int sms_per_tpc = 2;  // GB10 has 2 SMs per TPC

    // Scan TMD offsets in three modes
    const char* mode_names[] = {"", "mask only", "mask + enable bit", "mask + enable + ext"};

    for (int mode = 1; mode <= 3; mode++) {
        printf("=== Scan mode %d: %s ===\n", mode, mode_names[mode]);

        for (int offset = 0; offset <= 800; offset += 4) {
            g_scan_offset = offset;
            g_scan_mode = mode;

            read_smid_kernel<<<NUM_BLOCKS, 1024, 0, stream>>>(d_smids);
            cudaError_t err = cudaStreamSynchronize(stream);
            if (err != cudaSuccess) {
                // Kernel crashed - skip this offset
                cudaGetLastError();  // clear error
                continue;
            }

            cudaMemcpy(h_smids, d_smids, NUM_BLOCKS, cudaMemcpyDeviceToHost);
            int unique = count_unique_sms(h_smids, NUM_BLOCKS);
            int max_sm = get_max_sm(h_smids, NUM_BLOCKS);

            // If significantly fewer SMs than baseline, we found something
            if (unique <= sms_per_tpc && unique < baseline_unique) {
                printf("  ** HIT ** offset=0x%03x (%3d): %d unique SMs (max SM ID=%d)\n",
                       offset, offset, unique, max_sm);
            } else if (unique < baseline_unique / 2) {
                printf("  * partial* offset=0x%03x (%3d): %d unique SMs (max SM ID=%d)\n",
                       offset, offset, unique, max_sm);
            }
        }
        printf("\n");
    }

    g_scan_mode = 0;
    cudaFree(d_smids);
    cudaStreamDestroy(stream);
    free(h_smids);

    printf("Scan complete.\n");
    return 0;
}
