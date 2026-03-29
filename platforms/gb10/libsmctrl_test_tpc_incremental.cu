// Test that enabling TPC 1, 1-2, 1-3, 1-4 incrementally works correctly on GB10
#include <error.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <cuda_runtime.h>

#include "libsmctrl.h"
#include "testbench.h"

__global__ void read_smid(uint8_t* smid_arr) {
  if (threadIdx.x != 0)
    return;
  int smid;
  asm("mov.u32 %0, %%smid;" : "=r"(smid));
  smid_arr[blockIdx.x] = smid;
}

#define NUM_BLOCKS 142

static int sort_asc(const void* a, const void* b) {
  return *(uint8_t*)a - *(uint8_t*)b;
}

static int count_unique(uint8_t* arr, int len) {
  qsort(arr, len, 1, sort_asc);
  int num_uniq = 1;
  for (int i = 0; i < len - 1; i++)
    num_uniq += (arr[i] != arr[i + 1]);
  return num_uniq;
}

int main() {
  int num_sms, cap_major;
  uint32_t num_tpcs;
  int res;
  cudaStream_t stream;

  SAFE(cudaStreamCreate(&stream));
  SAFE(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0));
  SAFE(cudaDeviceGetAttribute(&cap_major, cudaDevAttrComputeCapabilityMajor, 0));
  if (res = libsmctrl_get_tpc_info_cuda(&num_tpcs, 0))
    error(1, res, "Unable to get TPC configuration");

  int sms_per_tpc = num_sms / num_tpcs;
  printf("GPU info: %d SMs, %d TPCs, %d SMs/TPC, compute %d.x\n\n",
         num_sms, num_tpcs, sms_per_tpc, cap_major);

  // Test enabling TPC 0..N for N = 0,1,2,3 (i.e. 1 TPC, 2 TPCs, 3 TPCs, 4 TPCs)
  int max_test_tpcs = 4;
  if ((uint32_t)max_test_tpcs > num_tpcs)
    max_test_tpcs = num_tpcs;

  int failed = 0;

  for (int num_enabled = 1; num_enabled <= max_test_tpcs; num_enabled++) {
    // Build mask: enable TPC 0..(num_enabled-1), disable the rest
    // mask bit set = TPC disabled
    uint128_t enable_bits = 0;
    for (int t = 0; t < num_enabled; t++) {
      uint128_t one = 1;
      enable_bits |= (one << t);
    }
    uint128_t mask = ~enable_bits;

    // Apply mask via global
    libsmctrl_set_global_mask(mask);

    uint8_t *smids_d, *smids_h;
    SAFE(cudaMalloc(&smids_d, NUM_BLOCKS));
    smids_h = (uint8_t*)malloc(NUM_BLOCKS);
    if (!smids_h) error(1, errno, "malloc failed");

    read_smid<<<NUM_BLOCKS, 1024, 0, stream>>>(smids_d);
    SAFE(cudaMemcpy(smids_h, smids_d, NUM_BLOCKS, cudaMemcpyDeviceToHost));

    int uniq = count_unique(smids_h, NUM_BLOCKS);
    int expected_sms = num_enabled * sms_per_tpc;

    printf("TPC 0..%d enabled (%d TPCs): used %d unique SMs (expected <= %d)\n",
           num_enabled - 1, num_enabled, uniq, expected_sms);

    // Print which SM IDs were used
    printf("  SM IDs used:");
    // re-run to get unsorted SM IDs for display (count_unique sorts)
    SAFE(cudaMemcpy(smids_h, smids_d, NUM_BLOCKS, cudaMemcpyDeviceToHost));
    qsort(smids_h, NUM_BLOCKS, 1, sort_asc);
    int prev = -1;
    for (int i = 0; i < NUM_BLOCKS; i++) {
      if (smids_h[i] != prev) {
        printf(" %d", smids_h[i]);
        prev = smids_h[i];
      }
    }
    printf("\n");

    if (uniq > expected_sms) {
      printf("  *** FAIL: used %d SMs but expected at most %d ***\n", uniq, expected_sms);
      failed++;
    } else {
      printf("  PASS\n");
    }
    printf("\n");

    SAFE(cudaFree(smids_d));
    free(smids_h);
  }

  // Reset global mask
  libsmctrl_set_global_mask(0);

  if (failed)
    printf("FAILED: %d test(s) failed\n", failed);
  else
    printf("ALL PASSED: TPC 1 through %d incremental tests passed\n", max_test_tpcs);

  return failed ? 1 : 0;
}
