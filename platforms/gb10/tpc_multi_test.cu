// Multi-TPC control verification test for GB10
// Tests various TPC combinations to verify mask works correctly.
//
// Build:
//   make libsmctrl.a
//   nvcc -ccbin g++ tpc_multi_test.cu -o tpc_multi_test -g -L. -l:libsmctrl.a -lcuda
// Usage:
//   ./tpc_multi_test [device_id]

#include <cuda.h>
#include <cuda_runtime.h>
#include <error.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "libsmctrl.h"
#include "testbench.h"

#define NUM_BLOCKS 256
#define MAX_SMS    256

__global__ void read_smid(uint8_t* out) {
    if (threadIdx.x != 0) return;
    int smid;
    asm("mov.u32 %0, %%smid;" : "=r"(smid));
    out[blockIdx.x] = smid;
}

static int count_unique(uint8_t* arr, int len, uint8_t* out_ids) {
    uint8_t tmp[NUM_BLOCKS];
    memcpy(tmp, arr, len);
    // sort
    for (int i = 0; i < len; i++)
        for (int j = i + 1; j < len; j++)
            if (tmp[i] > tmp[j]) { uint8_t t = tmp[i]; tmp[i] = tmp[j]; tmp[j] = t; }
    out_ids[0] = tmp[0];
    int n = 1;
    for (int i = 1; i < len; i++)
        if (tmp[i] != tmp[i - 1]) out_ids[n++] = tmp[i];
    return n;
}

static int run_masked(cudaStream_t stream, uint64_t mask, uint8_t* d_sm,
                      int* out_count, uint8_t* out_ids, const char* mask_type) {
    uint8_t h_sm[NUM_BLOCKS];

    if (strcmp(mask_type, "global") == 0) {
        libsmctrl_set_global_mask(mask);
    } else if (strcmp(mask_type, "next") == 0) {
        libsmctrl_set_next_mask(mask);
    } else if (strcmp(mask_type, "stream") == 0) {
        libsmctrl_set_stream_mask(stream, mask);
    }

    read_smid<<<NUM_BLOCKS, 1024, 0, stream>>>(d_sm);
    cudaError_t err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) {
        printf("  CUDA error: %s\n", cudaGetErrorString(err));
        return -1;
    }
    cudaMemcpy(h_sm, d_sm, NUM_BLOCKS, cudaMemcpyDeviceToHost);
    *out_count = count_unique(h_sm, NUM_BLOCKS, out_ids);

    if (strcmp(mask_type, "global") == 0) {
        libsmctrl_set_global_mask(0);
    }
    return 0;
}

int main(int argc, char** argv) {
    int dev = 0;
    if (argc > 1) dev = atoi(argv[1]);

    SAFE(cudaSetDevice(dev));

    int major, minor, num_sms;
    uint32_t num_tpcs;
    SAFE(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev));
    SAFE(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev));
    SAFE(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, dev));
    if (libsmctrl_get_tpc_info_cuda(&num_tpcs, dev))
        error(1, 0, "Failed to get TPC info");
    int sms_per_tpc = num_sms / num_tpcs;
    int ver;
    cuDriverGetVersion(&ver);

    printf("GPU: sm_%d%d, %d SMs, %u TPCs, %d SMs/TPC, CUDA %d.%d\n\n",
           major, minor, num_sms, num_tpcs, sms_per_tpc,
           ver / 1000, (ver % 1000) / 10);

    uint8_t *d_sm;
    SAFE(cudaMalloc(&d_sm, NUM_BLOCKS));
    cudaStream_t stream;
    SAFE(cudaStreamCreate(&stream));

    int total_pass = 0, total_fail = 0;
    int uniq;
    uint8_t ids[MAX_SMS];

    // ─── Test 1: Single TPC isolation (all TPCs individually) ───
    printf("=== Test 1: Single TPC Isolation ===\n");
    printf("  Enabling one TPC at a time and checking SM count = %d\n\n", sms_per_tpc);

    for (uint32_t tpc = 0; tpc < num_tpcs; tpc++) {
        uint64_t mask = ~(1ULL << tpc);  // disable all except this TPC
        run_masked(stream, mask, d_sm, &uniq, ids, "global");

        printf("  TPC %2u: %2d SMs [", tpc, uniq);
        for (int i = 0; i < uniq; i++) printf("%s%d", i ? "," : "", ids[i]);
        printf("]");

        if (uniq == sms_per_tpc) {
            printf(" PASS\n");
            total_pass++;
        } else {
            printf(" FAIL (expected %d)\n", sms_per_tpc);
            total_fail++;
        }
    }

    // ─── Test 2: TPC pairs ───
    printf("\n=== Test 2: TPC Pairs ===\n");
    printf("  Enabling two adjacent TPCs, expecting %d SMs each\n\n", sms_per_tpc * 2);

    int pair_tests[][2] = {
        {0, 1}, {2, 3}, {4, 5}, {10, 11}, {22, 23},
        {0, 12}, {5, 20}, {1, 23}  // non-adjacent pairs
    };
    int n_pairs = sizeof(pair_tests) / sizeof(pair_tests[0]);

    for (int p = 0; p < n_pairs; p++) {
        int a = pair_tests[p][0], b = pair_tests[p][1];
        if ((uint32_t)a >= num_tpcs || (uint32_t)b >= num_tpcs) continue;

        uint64_t mask = ~((1ULL << a) | (1ULL << b));
        run_masked(stream, mask, d_sm, &uniq, ids, "global");

        int expected = sms_per_tpc * 2;
        printf("  TPC {%d,%d}: %2d SMs [", a, b, uniq);
        for (int i = 0; i < uniq && i < 8; i++) printf("%s%d", i ? "," : "", ids[i]);
        if (uniq > 8) printf(",...");
        printf("]");

        if (uniq == expected) {
            printf(" PASS\n");
            total_pass++;
        } else {
            printf(" FAIL (expected %d)\n", expected);
            total_fail++;
        }
    }

    // ─── Test 3: Increasing TPC counts ───
    printf("\n=== Test 3: Scaling TPC Count (1, 2, 4, 8, 12, 16, 20, 24) ===\n");
    printf("  Enabling N TPCs from TPC 0, checking SM count = N * %d\n\n", sms_per_tpc);

    int counts[] = {1, 2, 4, 8, 12, 16, 20, 24};
    int n_counts = sizeof(counts) / sizeof(counts[0]);

    for (int c = 0; c < n_counts; c++) {
        int n = counts[c];
        if ((uint32_t)n > num_tpcs) continue;

        uint64_t enable_bits = (n >= 64) ? ~0ULL : ((1ULL << n) - 1);
        uint64_t mask = ~enable_bits;
        run_masked(stream, mask, d_sm, &uniq, ids, "global");

        int expected = sms_per_tpc * n;
        printf("  %2d TPCs: %2d SMs", n, uniq);

        if (uniq == expected) {
            printf(" PASS\n");
            total_pass++;
        } else {
            printf(" FAIL (expected %d)\n", expected);
            total_fail++;
        }
    }

    // ─── Test 4: Multi-TPC via next mask ───
    printf("\n=== Test 4: Multi-TPC via Next Mask ===\n");

    uint64_t mask_4tpc = ~0xFULL;  // enable TPC 0-3
    run_masked(stream, mask_4tpc, d_sm, &uniq, ids, "next");
    int expected_4 = sms_per_tpc * 4;
    printf("  Next mask (TPC 0-3): %d SMs [", uniq);
    for (int i = 0; i < uniq && i < 12; i++) printf("%s%d", i ? "," : "", ids[i]);
    if (uniq > 12) printf(",...");
    printf("]");
    if (uniq == expected_4) { printf(" PASS\n"); total_pass++; }
    else { printf(" FAIL (expected %d)\n", expected_4); total_fail++; }

    // Verify next mask was consumed
    libsmctrl_set_global_mask(0);
    run_masked(stream, 0, d_sm, &uniq, ids, "global");
    printf("  After consumed: %d SMs", uniq);
    if (uniq == num_sms) { printf(" PASS\n"); total_pass++; }
    else { printf(" FAIL (expected %d)\n", num_sms); total_fail++; }

    // ─── Test 5: Multi-TPC via stream mask ───
    printf("\n=== Test 5: Multi-TPC via Stream Mask ===\n");

    // 8 TPCs via stream
    uint64_t mask_8tpc = ~0xFFULL;
    run_masked(stream, mask_8tpc, d_sm, &uniq, ids, "stream");
    int expected_8 = sms_per_tpc * 8;
    printf("  Stream mask (TPC 0-7): %d SMs", uniq);
    if (uniq == expected_8) { printf(" PASS\n"); total_pass++; }
    else { printf(" FAIL (expected %d)\n", expected_8); total_fail++; }

    // Clear and verify
    libsmctrl_set_stream_mask(stream, 0);
    run_masked(stream, 0, d_sm, &uniq, ids, "global");
    printf("  After clear: %d SMs", uniq);
    if (uniq == num_sms) { printf(" PASS\n"); total_pass++; }
    else { printf(" FAIL (expected %d)\n", num_sms); total_fail++; }

    // ─── Test 6: Disjoint TPC sets on different streams ───
    printf("\n=== Test 6: Disjoint Streams ===\n");

    cudaStream_t stream_a, stream_b;
    SAFE(cudaStreamCreate(&stream_a));
    SAFE(cudaStreamCreate(&stream_b));

    uint64_t mask_a = ~0x00FULL;  // TPC 0-3
    uint64_t mask_b = ~0x0F0ULL;  // TPC 4-7
    libsmctrl_set_stream_mask(stream_a, mask_a);
    libsmctrl_set_stream_mask(stream_b, mask_b);

    uint8_t h_a[NUM_BLOCKS], h_b[NUM_BLOCKS];
    uint8_t *d_sm_a, *d_sm_b;
    SAFE(cudaMalloc(&d_sm_a, NUM_BLOCKS));
    SAFE(cudaMalloc(&d_sm_b, NUM_BLOCKS));

    read_smid<<<NUM_BLOCKS, 1024, 0, stream_a>>>(d_sm_a);
    read_smid<<<NUM_BLOCKS, 1024, 0, stream_b>>>(d_sm_b);
    SAFE(cudaStreamSynchronize(stream_a));
    SAFE(cudaStreamSynchronize(stream_b));
    SAFE(cudaMemcpy(h_a, d_sm_a, NUM_BLOCKS, cudaMemcpyDeviceToHost));
    SAFE(cudaMemcpy(h_b, d_sm_b, NUM_BLOCKS, cudaMemcpyDeviceToHost));

    uint8_t ids_a[MAX_SMS], ids_b[MAX_SMS];
    int count_a = count_unique(h_a, NUM_BLOCKS, ids_a);
    int count_b = count_unique(h_b, NUM_BLOCKS, ids_b);

    printf("  Stream A (TPC 0-3): %d SMs [", count_a);
    for (int i = 0; i < count_a && i < 10; i++) printf("%s%d", i ? "," : "", ids_a[i]);
    printf("]\n");
    printf("  Stream B (TPC 4-7): %d SMs [", count_b);
    for (int i = 0; i < count_b && i < 10; i++) printf("%s%d", i ? "," : "", ids_b[i]);
    printf("]\n");

    // Check no overlap
    int overlap = 0;
    for (int i = 0; i < count_a; i++)
        for (int j = 0; j < count_b; j++)
            if (ids_a[i] == ids_b[j]) overlap++;

    int expected_each = sms_per_tpc * 4;
    printf("  Overlap: %d SMs\n", overlap);
    printf("  Stream A count=%d (expect %d), Stream B count=%d (expect %d), overlap=%d (expect 0): ",
           count_a, expected_each, count_b, expected_each, overlap);
    if (count_a == expected_each && count_b == expected_each && overlap == 0) {
        printf("PASS\n");
        total_pass++;
    } else {
        printf("FAIL\n");
        total_fail++;
    }

    SAFE(cudaFree(d_sm_a));
    SAFE(cudaFree(d_sm_b));
    SAFE(cudaStreamDestroy(stream_a));
    SAFE(cudaStreamDestroy(stream_b));

    // ─── Summary ───
    printf("\n========================================\n");
    printf("Results: %d PASSED, %d FAILED\n", total_pass, total_fail);
    printf("========================================\n");

    SAFE(cudaFree(d_sm));
    SAFE(cudaStreamDestroy(stream));

    return total_fail > 0 ? 1 : 0;
}
