// Test enabling multiple TPCs simultaneously and see actual SM usage
#include <error.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
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

#define NUM_BLOCKS 256

static int sort_asc(const void* a, const void* b) {
  return *(uint8_t*)a - *(uint8_t*)b;
}

int main() {
  int num_sms;
  uint32_t num_tpcs;
  int res;
  cudaStream_t stream;

  SAFE(cudaStreamCreate(&stream));
  SAFE(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0));
  if (res = libsmctrl_get_tpc_info_cuda(&num_tpcs, 0))
    error(1, res, "Unable to get TPC configuration");

  int sms_per_tpc = num_sms / num_tpcs;
  printf("GPU: %d SMs, %d TPCs, %d SMs/TPC\n\n", num_sms, num_tpcs, sms_per_tpc);

  // Test: enable TPC 0..N-1 simultaneously
  for (int num_enabled = 1; num_enabled <= 8; num_enabled++) {
    uint128_t enable_bits = 0;
    for (int t = 0; t < num_enabled; t++)
      enable_bits |= ((uint128_t)1 << t);
    uint128_t mask = ~enable_bits;

    libsmctrl_set_stream_mask_ext(stream, mask);

    uint8_t *smids_d, *smids_h;
    SAFE(cudaMalloc(&smids_d, NUM_BLOCKS));
    smids_h = (uint8_t*)malloc(NUM_BLOCKS);

    read_smid<<<NUM_BLOCKS, 1024, 0, stream>>>(smids_d);
    SAFE(cudaMemcpy(smids_h, smids_d, NUM_BLOCKS, cudaMemcpyDeviceToHost));

    qsort(smids_h, NUM_BLOCKS, 1, sort_asc);

    printf("TPC 0..%d (%d TPCs) -> SMs:", num_enabled - 1, num_enabled);
    int prev = -1;
    int count = 0;
    for (int i = 0; i < NUM_BLOCKS; i++) {
      if (smids_h[i] != prev) {
        printf(" %d", smids_h[i]);
        prev = smids_h[i];
        count++;
      }
    }
    printf("  [%d SMs, expected %d]\n", count, num_enabled * sms_per_tpc);

    SAFE(cudaFree(smids_d));
    free(smids_h);
  }

  // Also test with global mask for comparison
  printf("\n--- Same test with libsmctrl_set_global_mask ---\n");
  // Reset stream mask
  libsmctrl_set_stream_mask_ext(stream, 0);

  for (int num_enabled = 1; num_enabled <= 8; num_enabled++) {
    uint128_t enable_bits = 0;
    for (int t = 0; t < num_enabled; t++)
      enable_bits |= ((uint128_t)1 << t);
    // global_mask only takes uint64_t
    uint64_t mask64 = ~(uint64_t)enable_bits;

    libsmctrl_set_global_mask(mask64);

    uint8_t *smids_d, *smids_h;
    SAFE(cudaMalloc(&smids_d, NUM_BLOCKS));
    smids_h = (uint8_t*)malloc(NUM_BLOCKS);

    read_smid<<<NUM_BLOCKS, 1024, 0, stream>>>(smids_d);
    SAFE(cudaMemcpy(smids_h, smids_d, NUM_BLOCKS, cudaMemcpyDeviceToHost));

    qsort(smids_h, NUM_BLOCKS, 1, sort_asc);

    printf("TPC 0..%d (%d TPCs) -> SMs:", num_enabled - 1, num_enabled);
    int prev = -1;
    int count = 0;
    for (int i = 0; i < NUM_BLOCKS; i++) {
      if (smids_h[i] != prev) {
        printf(" %d", smids_h[i]);
        prev = smids_h[i];
        count++;
      }
    }
    printf("  [%d SMs, expected %d]\n", count, num_enabled * sms_per_tpc);

    SAFE(cudaFree(smids_d));
    free(smids_h);
  }

  libsmctrl_set_global_mask(0);
  return 0;
}
