// Focused TMD offset scanner - tests specific candidate offsets
// Forks a child process for each test to avoid hangs
//
// Build:
//   nvcc -ccbin g++ tmd_scan2.cu -o tmd_scan2 -lcuda
// Usage:
//   ./tmd_scan2 [device_id]
//   ./tmd_scan2 0 --offset 0x130   # test a single offset
//   ./tmd_scan2 0 --range 0x120 0x160  # test a range

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <sys/wait.h>
#include <sys/types.h>

static const CUuuid callback_funcs_id = {
    0x2c, (char)0x8e, 0x0a, (char)0xd8,
    0x07, 0x10, (char)0xab, 0x4e,
    (char)0x90, (char)0xdd, 0x54, 0x71,
    (char)0x9f, (char)0xe5, (char)0xf7, 0x4b
};

#define QMD_DOMAIN 0xb
#define QMD_PRE_UPLOAD 0x1
#define NUM_BLOCKS 142

static int g_test_offset = -1;
static int g_test_mode = 0;  // 0=off, 1=mask, 2=mask+enable, 3=mask+enable+ext

__global__ void read_smid(uint8_t* out) {
    if (threadIdx.x != 0) return;
    int smid;
    asm("mov.u32 %0, %%smid;" : "=r"(smid));
    out[blockIdx.x] = smid;
}

static void test_callback(void *u, int domain, int cbid, const void *in_params) {
    if (g_test_mode == 0) return;
    void* tmd = *((void**)in_params + 4);
    if (!tmd) return;

    uint32_t lower = 0xFFFFFFFE;  // enable only TPC 0
    uint32_t upper = 0xFFFFFFFF;

    int off = g_test_offset;

    if (g_test_mode >= 1) {
        *(uint32_t*)((uint8_t*)tmd + off) = lower;
        *(uint32_t*)((uint8_t*)tmd + off + 4) = upper;
    }
    if (g_test_mode >= 2) {
        *(uint32_t*)tmd |= 0x80000000;
    }
    if (g_test_mode >= 3) {
        *(uint32_t*)((uint8_t*)tmd + off + 8) = 0xFFFFFFFF;
        *(uint32_t*)((uint8_t*)tmd + off + 12) = 0xFFFFFFFF;
    }
}

static int sort_u8(const void* a, const void* b) {
    return *(uint8_t*)a - *(uint8_t*)b;
}

// Run one test: returns unique SM count, or -1 on error
static int run_test(int dev, int offset, int mode) {
    cudaSetDevice(dev);

    uintptr_t* tbl;
    cuGetExportTable((const void**)&tbl, &callback_funcs_id);
    auto sub = (int(*)(uint32_t*, void(*)(void*,int,int,const void*), void*))*(tbl+3);
    auto en  = (int(*)(uint32_t,uint32_t,int,int))*(tbl+6);

    uint32_t h;
    sub(&h, test_callback, NULL);
    en(1, h, QMD_DOMAIN, QMD_PRE_UPLOAD);

    uint8_t *d_sm, *h_sm;
    cudaMalloc(&d_sm, NUM_BLOCKS);
    h_sm = (uint8_t*)malloc(NUM_BLOCKS);

    cudaStream_t s;
    cudaStreamCreate(&s);

    g_test_offset = offset;
    g_test_mode = mode;

    read_smid<<<NUM_BLOCKS, 1024, 0, s>>>(d_sm);
    cudaError_t err = cudaStreamSynchronize(s);
    if (err != cudaSuccess) {
        cudaGetLastError();
        free(h_sm);
        return -1;
    }

    cudaMemcpy(h_sm, d_sm, NUM_BLOCKS, cudaMemcpyDeviceToHost);

    // Count unique SMs
    qsort(h_sm, NUM_BLOCKS, 1, sort_u8);
    int uniq = 1;
    uint8_t max_sm = h_sm[0];
    for (int i = 1; i < NUM_BLOCKS; i++) {
        if (h_sm[i] != h_sm[i-1]) uniq++;
        if (h_sm[i] > max_sm) max_sm = h_sm[i];
    }

    // Print SM list if constrained
    if (uniq <= 4) {
        printf("    SMs used:");
        uint8_t prev = 255;
        for (int i = 0; i < NUM_BLOCKS; i++) {
            if (h_sm[i] != prev) { printf(" %d", h_sm[i]); prev = h_sm[i]; }
        }
        printf("\n");
    }

    cudaFree(d_sm);
    cudaStreamDestroy(s);
    free(h_sm);
    return uniq;
}

// Fork-safe test: runs in child process with timeout
static int safe_test(int dev, int offset, int mode, int timeout_sec) {
    int pipefd[2];
    pipe(pipefd);

    pid_t pid = fork();
    if (pid == 0) {
        // Child
        close(pipefd[0]);
        int result = run_test(dev, offset, mode);
        write(pipefd[1], &result, sizeof(result));
        close(pipefd[1]);
        _exit(0);
    }

    // Parent
    close(pipefd[1]);

    // Wait with timeout
    int status;
    int elapsed = 0;
    while (elapsed < timeout_sec * 10) {
        pid_t w = waitpid(pid, &status, WNOHANG);
        if (w > 0) break;
        usleep(100000);  // 100ms
        elapsed++;
    }

    if (elapsed >= timeout_sec * 10) {
        kill(pid, SIGKILL);
        waitpid(pid, &status, 0);
        close(pipefd[0]);
        return -2;  // timeout
    }

    int result = -1;
    read(pipefd[0], &result, sizeof(result));
    close(pipefd[0]);
    return result;
}

int main(int argc, char** argv) {
    int dev = 0;
    int single_offset = -1;
    int range_start = -1, range_end = -1;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--offset") && i+1 < argc) {
            single_offset = strtol(argv[++i], NULL, 0);
        } else if (!strcmp(argv[i], "--range") && i+2 < argc) {
            range_start = strtol(argv[++i], NULL, 0);
            range_end = strtol(argv[++i], NULL, 0);
        } else {
            dev = atoi(argv[i]);
        }
    }

    int major, minor, num_sms;
    cudaSetDevice(dev);
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
    cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev);
    cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, dev);
    int cuda_ver;
    cuDriverGetVersion(&cuda_ver);
    printf("GPU: sm_%d%d, %d SMs, CUDA %d.%d\n", major, minor, num_sms,
           cuda_ver / 1000, (cuda_ver % 1000) / 10);

    // Baseline
    printf("\nBaseline (no modification):\n");
    int baseline = safe_test(dev, 0, 0, 5);
    printf("  %d unique SMs\n\n", baseline);

    if (single_offset >= 0) {
        // Test single offset in all modes
        printf("Testing offset 0x%x:\n", single_offset);
        for (int mode = 1; mode <= 3; mode++) {
            const char* names[] = {"", "mask", "mask+enable", "mask+enable+ext"};
            printf("  mode %d (%s): ", mode, names[mode]);
            fflush(stdout);
            int r = safe_test(dev, single_offset, mode, 5);
            if (r == -2) printf("TIMEOUT\n");
            else if (r == -1) printf("ERROR\n");
            else printf("%d unique SMs%s\n", r, r <= 2 ? " *** HIT ***" : "");
        }
        return 0;
    }

    // Define candidate offsets to test
    // Based on TMD dump analysis and known patterns
    int candidates[] = {
        // Hopper V04_00 known offsets
        84, 88,     // V01_06 mask (Kepler-Pascal)
        304, 308,   // V04_00 mask (Hopper)

        // Near the 0x000fffff at offset 0x278
        0x270, 0x274, 0x278, 0x27c, 0x280,

        // Systematic scan of interesting TMD regions
        // Near beginning (version/enable area)
        0x00, 0x04, 0x08, 0x0c, 0x10,
        // After version area
        0x30, 0x34, 0x38, 0x3c, 0x40, 0x44, 0x48, 0x4c,
        // Extended range after Hopper offsets
        0x140, 0x144, 0x148, 0x14c,
        0x150, 0x154, 0x158, 0x15c,
        0x160, 0x164, 0x168, 0x16c,
        0x170, 0x174, 0x178, 0x17c,
        0x180, 0x184, 0x188, 0x18c,
        // Around 0x200-0x2C0 area (new TMD might be larger)
        0x200, 0x204, 0x208, 0x20c,
        0x210, 0x214, 0x218, 0x21c,
        0x220, 0x224, 0x228, 0x22c,
        0x230, 0x234, 0x238, 0x23c,
        0x240, 0x244, 0x248, 0x24c,
        0x250, 0x254, 0x258, 0x25c,
        0x260, 0x264, 0x268, 0x26c,

        // Post-mask area
        0x2a0, 0x2a4, 0x2a8, 0x2ac,
        0x2b0, 0x2b4, 0x2b8, 0x2bc,
        0x2c0, 0x2c4, 0x2c8, 0x2cc,

        // 0x300+ area
        0x300, 0x304, 0x308, 0x30c,
        0x310, 0x314, 0x318, 0x31c,
        0x320, 0x324, 0x328, 0x32c,
        0x330, 0x334, 0x338, 0x33c,
        0x340, 0x344, 0x348, 0x34c,
        0x350, 0x354, 0x358, 0x35c,

        -1  // sentinel
    };

    if (range_start >= 0 && range_end >= 0) {
        // Override with range scan
        printf("Scanning range 0x%x - 0x%x\n", range_start, range_end);
        for (int off = range_start; off <= range_end; off += 4) {
            for (int mode = 1; mode <= 3; mode++) {
                printf("  0x%03x mode=%d: ", off, mode);
                fflush(stdout);
                int r = safe_test(dev, off, mode, 3);
                if (r == -2) printf("TIMEOUT\n");
                else if (r == -1) printf("ERROR\n");
                else {
                    printf("%d SMs", r);
                    if (r <= 2) printf(" *** HIT ***");
                    else if (r < baseline / 2) printf(" * partial *");
                    printf("\n");
                }
            }
        }
        return 0;
    }

    // Test all candidate offsets
    printf("Testing %d candidate offsets (3 modes each)...\n\n", (int)(sizeof(candidates)/sizeof(candidates[0]) - 1));

    for (int i = 0; candidates[i] >= 0; i++) {
        int off = candidates[i];
        for (int mode = 1; mode <= 3; mode++) {
            int r = safe_test(dev, off, mode, 3);
            if (r == -2) {
                printf("  0x%03x mode=%d: TIMEOUT\n", off, mode);
            } else if (r == -1) {
                // silent skip for errors
            } else if (r <= 2) {
                printf("  ** HIT **  0x%03x (%3d) mode=%d: %d unique SMs\n", off, off, mode, r);
            } else if (r < baseline / 2) {
                printf("  * partial* 0x%03x (%3d) mode=%d: %d unique SMs\n", off, off, mode, r);
            }
        }
    }

    printf("\nDone.\n");
    return 0;
}
