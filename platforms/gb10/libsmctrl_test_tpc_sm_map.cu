// Map all TPC-to-SM assignments on the GPU
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

#define NUM_BLOCKS 142

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

  printf("%-6s -> SM IDs\n", "TPC");
  printf("-------------------------------\n");

  // For each TPC, enable only that TPC and see which SMs are used
  for (uint32_t tpc = 0; tpc < num_tpcs; tpc++) {
    uint128_t enable_bit = (uint128_t)1 << tpc;
    uint128_t mask = ~enable_bit;

    libsmctrl_set_stream_mask_ext(stream, mask);

    uint8_t *smids_d, *smids_h;
    SAFE(cudaMalloc(&smids_d, NUM_BLOCKS));
    smids_h = (uint8_t*)malloc(NUM_BLOCKS);

    read_smid<<<NUM_BLOCKS, 1024, 0, stream>>>(smids_d);
    SAFE(cudaMemcpy(smids_h, smids_d, NUM_BLOCKS, cudaMemcpyDeviceToHost));

    qsort(smids_h, NUM_BLOCKS, 1, sort_asc);

    printf("TPC %2d ->", tpc);
    int prev = -1;
    for (int i = 0; i < NUM_BLOCKS; i++) {
      if (smids_h[i] != prev) {
        printf(" %2d", smids_h[i]);
        prev = smids_h[i];
      }
    }
    printf("\n");

    SAFE(cudaFree(smids_d));
    free(smids_h);
  }

  // Reset
  libsmctrl_set_stream_mask_ext(stream, 0);

  // Print grouping analysis
  printf("\n--- Grouping Analysis (stride pattern) ---\n");
  printf("If GPCs each own 8 SM slots, the mapping should show\n");
  printf("TPCs cycling through GPCs with stride 8 in SM IDs.\n");

  return 0;
}
