// TMD structure dump utility for reverse engineering new TMD layouts.
// Hooks the CUDA TMD callback and dumps raw bytes around known offset areas.
//
// Build:
//   nvcc -ccbin g++ tmd_dump.cu -o tmd_dump -lcuda -I/usr/local/cuda/include -L/usr/local/cuda/lib64
// Usage:
//   ./tmd_dump [device_id]

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

// Callback function table UUID (same as libsmctrl)
static const CUuuid callback_funcs_id = {
    0x2c, (char)0x8e, 0x0a, (char)0xd8,
    0x07, 0x10, (char)0xab, 0x4e,
    (char)0x90, (char)0xdd, 0x54, 0x71,
    (char)0x9f, (char)0xe5, (char)0xf7, 0x4b
};

#define QMD_DOMAIN 0xb
#define QMD_PRE_UPLOAD 0x1

static int dump_count = 0;

static void tmd_dump_callback(void *ukwn, int domain, int cbid, const void *in_params) {
    if (dump_count > 0) return;  // Only dump once
    dump_count++;

    uint32_t param_size = *(uint32_t*)in_params;
    printf("=== TMD Dump (domain=%d, cbid=%d) ===\n", domain, cbid);
    printf("in_params size field: 0x%x (%u bytes)\n", param_size, param_size);

    // Dump in_params pointers
    int num_ptrs = param_size / sizeof(void*);
    printf("in_params has %d pointer-sized entries:\n", num_ptrs);
    for (int i = 0; i < num_ptrs && i < 16; i++) {
        void* ptr = *((void**)in_params + i);
        printf("  [%2d] = %p\n", i, ptr);
    }

    // Get TMD pointer (index 4, same as libsmctrl)
    if (param_size < 5 * sizeof(void*)) {
        printf("ERROR: in_params too small for TMD pointer (size=0x%x, need 0x%lx)\n",
               param_size, 5 * sizeof(void*));
        return;
    }

    void* tmd = *((void**)in_params + 4);
    if (!tmd) {
        printf("ERROR: TMD pointer is NULL\n");
        return;
    }
    printf("\nTMD pointer: %p\n", tmd);

    // Dump first 512 bytes of TMD as hex
    uint8_t* tmd_bytes = (uint8_t*)tmd;
    printf("\n=== TMD Raw Dump (first 512 bytes) ===\n");
    printf("Offset  | 00 01 02 03 04 05 06 07  08 09 0A 0B 0C 0D 0E 0F | ASCII\n");
    printf("--------+--------------------------------------------------+-----------------\n");

    for (int row = 0; row < 512; row += 16) {
        printf("0x%04x  | ", row);
        for (int col = 0; col < 16; col++) {
            printf("%02x ", tmd_bytes[row + col]);
            if (col == 7) printf(" ");
        }
        printf("| ");
        for (int col = 0; col < 16; col++) {
            uint8_t c = tmd_bytes[row + col];
            printf("%c", (c >= 0x20 && c < 0x7f) ? c : '.');
        }
        printf("\n");
    }

    // Highlight known offset areas
    printf("\n=== Key Offsets Analysis ===\n");

    // TMD version field (traditionally at offset 72 = 0x48)
    printf("Traditional TMD version offset (0x48/72): 0x%02x\n", tmd_bytes[72]);

    // Check nearby bytes for version-like values
    printf("\nScanning for version-like values (0x16, 0x40, 0x50+):\n");
    for (int i = 0; i < 512; i++) {
        if (tmd_bytes[i] == 0x16 || tmd_bytes[i] == 0x40 ||
            (tmd_bytes[i] >= 0x50 && tmd_bytes[i] <= 0x60)) {
            printf("  offset 0x%03x (%3d): 0x%02x", i, i, tmd_bytes[i]);
            // Show surrounding context
            printf("  context: ");
            for (int j = (i > 3 ? i - 3 : 0); j < i + 4 && j < 512; j++) {
                if (j == i) printf("[%02x]", tmd_bytes[j]);
                else printf(" %02x ", tmd_bytes[j]);
            }
            printf("\n");
        }
    }

    // Dump as 32-bit words for mask analysis
    printf("\n=== TMD as 32-bit words (first 512 bytes) ===\n");
    uint32_t* tmd_words = (uint32_t*)tmd;
    for (int i = 0; i < 128; i++) {
        if (i % 4 == 0) printf("0x%04x: ", i * 4);
        printf("0x%08x ", tmd_words[i]);
        if (i % 4 == 3) printf("\n");
    }

    // Known Hopper V04_00 offsets for comparison
    printf("\n=== Hopper V04_00 reference offsets ===\n");
    printf("  TMD+0    (enable bit area): 0x%08x\n", tmd_words[0]);
    printf("  TMD+72   (version field):   0x%02x\n", tmd_bytes[72]);
    printf("  TMD+304  (lower mask):      0x%08x\n", *(uint32_t*)(tmd_bytes + 304));
    printf("  TMD+308  (upper mask):      0x%08x\n", *(uint32_t*)(tmd_bytes + 308));
    printf("  TMD+312  (ext lower):       0x%08x\n", *(uint32_t*)(tmd_bytes + 312));
    printf("  TMD+316  (ext upper):       0x%08x\n", *(uint32_t*)(tmd_bytes + 316));

    // Scan for potential mask fields (look for 0xFFFFFFFF or 0x00FFFFFF patterns)
    printf("\n=== Scanning for mask-like 32-bit values (0xFFFFFFFF, 0x00FFFFFF, 0x00000000 clusters) ===\n");
    for (int i = 0; i < 128; i++) {
        uint32_t w = tmd_words[i];
        if (w == 0xFFFFFFFF || w == 0x00FFFFFF || w == 0xFFFF0000 ||
            (w != 0 && (w & 0xFFFF0000) == 0xFFFF0000) ||
            (w != 0 && (w & 0x0000FFFF) == 0x0000FFFF)) {
            printf("  word[%3d] (offset 0x%03x): 0x%08x\n", i, i * 4, w);
        }
    }

    // Extended dump for larger TMD (up to 1024 bytes)
    printf("\n=== TMD Extended Dump (512-1024 bytes) ===\n");
    printf("Offset  | 00 01 02 03 04 05 06 07  08 09 0A 0B 0C 0D 0E 0F\n");
    printf("--------+--------------------------------------------------\n");
    for (int row = 512; row < 1024; row += 16) {
        printf("0x%04x  | ", row);
        for (int col = 0; col < 16; col++) {
            printf("%02x ", tmd_bytes[row + col]);
            if (col == 7) printf(" ");
        }
        printf("\n");
    }

    printf("\n=== TMD extended as 32-bit words (512-1024) ===\n");
    for (int i = 128; i < 256; i++) {
        if (i % 4 == 0) printf("0x%04x: ", i * 4);
        printf("0x%08x ", tmd_words[i]);
        if (i % 4 == 3) printf("\n");
    }

    printf("\n=== Dump complete ===\n");
}

__global__ void dummy_kernel(int* out) {
    if (threadIdx.x == 0)
        out[blockIdx.x] = blockIdx.x;
}

int main(int argc, char** argv) {
    int dev = 0;
    if (argc > 1) dev = atoi(argv[1]);

    cudaSetDevice(dev);

    // Print GPU info
    int major, minor, num_sms;
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
    cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev);
    cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, dev);
    int cuda_ver;
    cuDriverGetVersion(&cuda_ver);

    printf("GPU: sm_%d%d, %d SMs, CUDA %d.%d\n\n", major, minor, num_sms,
           cuda_ver / 1000, (cuda_ver % 1000) / 10);

    // Set up callback
    uintptr_t* tbl_base;
    cuGetExportTable((const void**)&tbl_base, &callback_funcs_id);
    uintptr_t subscribe_addr = *(tbl_base + 3);
    uintptr_t enable_addr = *(tbl_base + 6);

    int (*subscribe)(uint32_t*, void(*)(void*, int, int, const void*), void*) =
        (int (*)(uint32_t*, void(*)(void*, int, int, const void*), void*))subscribe_addr;
    int (*enable)(uint32_t, uint32_t, int, int) =
        (int (*)(uint32_t, uint32_t, int, int))enable_addr;

    uint32_t hndl;
    int res = subscribe(&hndl, tmd_dump_callback, NULL);
    if (res) { printf("subscribe failed: %d\n", res); return 1; }
    res = enable(1, hndl, QMD_DOMAIN, QMD_PRE_UPLOAD);
    if (res) { printf("enable failed: %d\n", res); return 1; }

    printf("Callback registered, launching kernel...\n\n");

    // Launch a simple kernel to trigger the callback
    int* d_out;
    cudaMalloc(&d_out, 64 * sizeof(int));
    dummy_kernel<<<64, 32>>>(d_out);
    cudaDeviceSynchronize();
    cudaFree(d_out);

    return 0;
}
