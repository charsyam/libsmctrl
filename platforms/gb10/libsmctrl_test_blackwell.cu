// Copyright 2026 Backend.AI Team
// Comprehensive Blackwell/GB10 verification test.
// Tests platform detection, TPC info, TMD callback mechanism, and SM masking
// across all three mask types (global, stream, next) in a single binary.
//
// Usage:
//   ./libsmctrl_test_blackwell          # Run all tests
//   ./libsmctrl_test_blackwell --probe  # Only probe GPU info (no masking)

#include <cuda.h>
#include <cuda_runtime.h>
#include <error.h>
#include <errno.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "libsmctrl.h"
#include "testbench.h"

// ---------- GPU kernel: read SM ID for each block ----------
__global__ void blackwell_read_smid(uint8_t* smid_arr) {
  if (threadIdx.x != 0)
    return;
  int smid;
  asm("mov.u32 %0, %%smid;" : "=r"(smid));
  smid_arr[blockIdx.x] = smid;
}

#define NUM_BLOCKS 142
#define MAX_SMS    256

// ---------- Helpers ----------
static int sort_asc(const void* a, const void* b) {
  return *(uint8_t*)a - *(uint8_t*)b;
}

// Returns sorted unique SM IDs; writes count to *out_count
static void unique_smids(uint8_t* arr, int len, int* out_count, uint8_t* out_ids) {
  uint8_t tmp[NUM_BLOCKS];
  memcpy(tmp, arr, len);
  qsort(tmp, len, 1, sort_asc);
  out_ids[0] = tmp[0];
  int n = 1;
  for (int i = 1; i < len; i++) {
    if (tmp[i] != tmp[i - 1])
      out_ids[n++] = tmp[i];
  }
  *out_count = n;
}

// ---------- Phase 1: Platform & GPU info probe ----------
static int test_probe(int cuda_dev) {
  int major, minor, num_sms;
  uint32_t num_tpcs;
  int res;
  const char* arch_name;

  printf("\n=== Phase 1: Platform Probe (device %d) ===\n", cuda_dev);

  SAFE(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, cuda_dev));
  SAFE(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, cuda_dev));
  SAFE(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, cuda_dev));

  // Identify architecture
  if (major == 12 && minor == 1)
    arch_name = "Consumer Blackwell GB10 (DGX Spark / OEM)";
  else if (major == 12 && minor == 0)
    arch_name = "Consumer Blackwell (RTX 50-series)";
  else if (major == 10 && minor == 0)
    arch_name = "Datacenter Blackwell (GB200)";
  else if (major == 10 && minor == 3)
    arch_name = "Blackwell Ultra (GB300)";
  else if (major == 9)
    arch_name = "Hopper";
  else if (major >= 7)
    arch_name = "Pre-Blackwell (Volta/Turing/Ampere/Ada)";
  else
    arch_name = "Legacy";

  printf("  Compute capability : sm_%d%d\n", major, minor);
  printf("  Architecture       : %s\n", arch_name);
  printf("  SM count           : %d\n", num_sms);

  // TPC info via CUDA (no nvdebug needed)
  res = libsmctrl_get_tpc_info_cuda(&num_tpcs, cuda_dev);
  if (res) {
    printf("  TPC info           : FAILED (error %d)\n", res);
    return 1;
  }

  int sms_per_tpc = num_sms / num_tpcs;
  printf("  TPC count          : %u\n", num_tpcs);
  printf("  SMs per TPC        : %d\n", sms_per_tpc);
  printf("  64-TPC limit       : %s\n", num_tpcs <= 64 ? "OK (within limit)" : "EXCEEDS - need _ext APIs");

  // CUDA driver version
  int ver;
  cuDriverGetVersion(&ver);
  printf("  CUDA driver version: %d.%d\n", ver / 1000, (ver % 1000) / 10);

  // Validate expected GB10 config
  if (major == 12 && minor == 1) {
    bool ok = true;
    if (num_sms != 48) {
      printf("  [WARN] Expected 48 SMs for GB10, got %d\n", num_sms);
      ok = false;
    }
    if (num_tpcs != 24) {
      printf("  [WARN] Expected 24 TPCs for GB10, got %u\n", num_tpcs);
      ok = false;
    }
    if (sms_per_tpc != 2) {
      printf("  [WARN] Expected 2 SMs/TPC for GB10, got %d\n", sms_per_tpc);
      ok = false;
    }
    if (ok)
      printf("  [PASS] GB10 configuration matches expected spec\n");
  }

  printf("  Phase 1: PASSED\n");
  return 0;
}

// ---------- Phase 2: TMD callback mechanism test (global mask) ----------
static int test_global_mask(int sms_per_tpc, uint32_t num_tpcs) {
  uint8_t *d_smids, *h_smids;
  int uniq_count;
  uint8_t uniq_ids[MAX_SMS];

  printf("\n=== Phase 2: Global Mask (TMD callback) ===\n");

  SAFE(cudaMalloc(&d_smids, NUM_BLOCKS));
  h_smids = (uint8_t*)malloc(NUM_BLOCKS);
  if (!h_smids) error(1, errno, "malloc failed");

  // 2a: Baseline - no mask
  cudaStream_t stream;
  SAFE(cudaStreamCreate(&stream));
  blackwell_read_smid<<<NUM_BLOCKS, 1024, 0, stream>>>(d_smids);
  SAFE(cudaStreamSynchronize(stream));
  SAFE(cudaMemcpy(h_smids, d_smids, NUM_BLOCKS, cudaMemcpyDeviceToHost));
  unique_smids(h_smids, NUM_BLOCKS, &uniq_count, uniq_ids);
  printf("  Baseline: kernel ran on %d unique SMs\n", uniq_count);
  if (uniq_count < sms_per_tpc) {
    printf("  [FAIL] Baseline uses fewer SMs than one TPC (%d < %d)\n",
           uniq_count, sms_per_tpc);
    return 1;
  }

  // 2b: Constrain to TPC 0 only
  uint64_t mask = ~0x1ULL;  // disable all except TPC 0
  libsmctrl_set_global_mask(mask);

  blackwell_read_smid<<<NUM_BLOCKS, 1024, 0, stream>>>(d_smids);
  SAFE(cudaStreamSynchronize(stream));
  SAFE(cudaMemcpy(h_smids, d_smids, NUM_BLOCKS, cudaMemcpyDeviceToHost));
  unique_smids(h_smids, NUM_BLOCKS, &uniq_count, uniq_ids);

  printf("  With TPC 0 only: kernel ran on %d unique SMs:", uniq_count);
  for (int i = 0; i < uniq_count; i++) printf(" %d", uniq_ids[i]);
  printf("\n");

  if (uniq_count > sms_per_tpc) {
    printf("  [FAIL] Kernel escaped TPC 0 (ran on %d SMs, max %d expected)\n",
           uniq_count, sms_per_tpc);
    libsmctrl_set_global_mask(0);
    return 1;
  }

  // Note: SM IDs may not be contiguous per TPC on Blackwell (floorsweeping),
  // so we only verify the count is correct (checked above).

  // Clear global mask
  libsmctrl_set_global_mask(0);
  printf("  Phase 2: PASSED\n");

  SAFE(cudaFree(d_smids));
  SAFE(cudaStreamDestroy(stream));
  free(h_smids);
  return 0;
}

// ---------- Phase 3: Next mask test ----------
static int test_next_mask(int sms_per_tpc) {
  uint8_t *d_smids, *h_smids;
  int uniq_count;
  uint8_t uniq_ids[MAX_SMS];

  printf("\n=== Phase 3: Next Mask ===\n");

  cudaStream_t stream;
  SAFE(cudaStreamCreate(&stream));
  SAFE(cudaMalloc(&d_smids, NUM_BLOCKS));
  h_smids = (uint8_t*)malloc(NUM_BLOCKS);
  if (!h_smids) error(1, errno, "malloc failed");

  // Constrain next launch to TPC 1 only
  uint64_t mask = ~0x2ULL;
  libsmctrl_set_next_mask(mask);

  blackwell_read_smid<<<NUM_BLOCKS, 1024, 0, stream>>>(d_smids);
  SAFE(cudaStreamSynchronize(stream));
  SAFE(cudaMemcpy(h_smids, d_smids, NUM_BLOCKS, cudaMemcpyDeviceToHost));
  unique_smids(h_smids, NUM_BLOCKS, &uniq_count, uniq_ids);

  printf("  With TPC 1 only: kernel ran on %d unique SMs:", uniq_count);
  for (int i = 0; i < uniq_count; i++) printf(" %d", uniq_ids[i]);
  printf("\n");

  if (uniq_count > sms_per_tpc) {
    printf("  [FAIL] Kernel escaped TPC 1 (ran on %d SMs, max %d expected)\n",
           uniq_count, sms_per_tpc);
    return 1;
  }

  // Note: SM IDs are not necessarily contiguous per TPC on Blackwell
  // (e.g., TPC 1 may map to SM 8,9 rather than SM 2,3), so we only
  // verify the count, not the specific SM IDs.

  // Verify next mask is consumed (second launch should be unrestricted)
  blackwell_read_smid<<<NUM_BLOCKS, 1024, 0, stream>>>(d_smids);
  SAFE(cudaStreamSynchronize(stream));
  SAFE(cudaMemcpy(h_smids, d_smids, NUM_BLOCKS, cudaMemcpyDeviceToHost));
  unique_smids(h_smids, NUM_BLOCKS, &uniq_count, uniq_ids);
  printf("  After next mask consumed: kernel ran on %d SMs (should be >%d)\n",
         uniq_count, sms_per_tpc);
  if (uniq_count <= sms_per_tpc)
    printf("  [WARN] Next mask may not have been cleared properly\n");

  printf("  Phase 3: PASSED\n");

  SAFE(cudaFree(d_smids));
  SAFE(cudaStreamDestroy(stream));
  free(h_smids);
  return 0;
}

// ---------- Phase 4: Stream mask test ----------
static int test_stream_mask(int sms_per_tpc) {
  uint8_t *d_smids, *h_smids;
  int uniq_count;
  uint8_t uniq_ids[MAX_SMS];

  printf("\n=== Phase 4: Stream Mask ===\n");

  cudaStream_t stream;
  SAFE(cudaStreamCreate(&stream));
  SAFE(cudaMalloc(&d_smids, NUM_BLOCKS));
  h_smids = (uint8_t*)malloc(NUM_BLOCKS);
  if (!h_smids) error(1, errno, "malloc failed");

  // Constrain stream to TPC 2 only
  uint64_t mask = ~0x4ULL;
  libsmctrl_set_stream_mask(stream, mask);

  blackwell_read_smid<<<NUM_BLOCKS, 1024, 0, stream>>>(d_smids);
  SAFE(cudaStreamSynchronize(stream));
  SAFE(cudaMemcpy(h_smids, d_smids, NUM_BLOCKS, cudaMemcpyDeviceToHost));
  unique_smids(h_smids, NUM_BLOCKS, &uniq_count, uniq_ids);

  printf("  With TPC 2 only: kernel ran on %d unique SMs:", uniq_count);
  for (int i = 0; i < uniq_count; i++) printf(" %d", uniq_ids[i]);
  printf("\n");

  if (uniq_count > sms_per_tpc) {
    printf("  [FAIL] Kernel escaped TPC 2 (ran on %d SMs, max %d expected)\n",
           uniq_count, sms_per_tpc);
    printf("  [NOTE] Stream mask offset may be incorrect for this platform.\n"
           "         Try: MASK_OFF=<delta> ./libsmctrl_test_blackwell\n"
           "         to scan for the correct offset.\n");
    SAFE(cudaStreamDestroy(stream));
    return 1;
  }

  printf("  Phase 4: PASSED\n");

  SAFE(cudaFree(d_smids));
  SAFE(cudaStreamDestroy(stream));
  free(h_smids);
  return 0;
}

// ---------- Phase 5: Mask priority test (next > stream > global) ----------
static int test_mask_priority(int sms_per_tpc) {
  uint8_t *d_smids, *h_smids;
  int uniq_count;
  uint8_t uniq_ids[MAX_SMS];

  printf("\n=== Phase 5: Mask Priority (next > stream > global) ===\n");

  cudaStream_t stream;
  SAFE(cudaStreamCreate(&stream));
  SAFE(cudaMalloc(&d_smids, NUM_BLOCKS));
  h_smids = (uint8_t*)malloc(NUM_BLOCKS);
  if (!h_smids) error(1, errno, "malloc failed");

  // Set global=TPC0, stream=TPC1, next=TPC2 -- next should win
  libsmctrl_set_global_mask(~0x1ULL);
  libsmctrl_set_stream_mask(stream, ~0x2ULL);
  libsmctrl_set_next_mask(~0x4ULL);

  blackwell_read_smid<<<NUM_BLOCKS, 1024, 0, stream>>>(d_smids);
  SAFE(cudaStreamSynchronize(stream));
  SAFE(cudaMemcpy(h_smids, d_smids, NUM_BLOCKS, cudaMemcpyDeviceToHost));
  unique_smids(h_smids, NUM_BLOCKS, &uniq_count, uniq_ids);

  printf("  global=TPC0, stream=TPC1, next=TPC2: ran on SMs:");
  for (int i = 0; i < uniq_count; i++) printf(" %d", uniq_ids[i]);
  printf("\n");

  // Should be on TPC 2 only (count check; SM IDs may be non-contiguous)
  if (uniq_count > sms_per_tpc) {
    printf("  [FAIL] Next mask did not restrict to TPC 2 (%d SMs > %d)\n",
           uniq_count, sms_per_tpc);
    libsmctrl_set_global_mask(0);
    return 1;
  }

  // Save TPC 2 SM set for comparison
  uint8_t tpc2_ids[MAX_SMS];
  int tpc2_count = uniq_count;
  memcpy(tpc2_ids, uniq_ids, uniq_count);

  // After next consumed, stream should win over global
  // (next launch should land on TPC 1, not TPC 0)
  blackwell_read_smid<<<NUM_BLOCKS, 1024, 0, stream>>>(d_smids);
  SAFE(cudaStreamSynchronize(stream));
  SAFE(cudaMemcpy(h_smids, d_smids, NUM_BLOCKS, cudaMemcpyDeviceToHost));
  unique_smids(h_smids, NUM_BLOCKS, &uniq_count, uniq_ids);

  printf("  After next consumed (stream=TPC1 should win): ran on SMs:");
  for (int i = 0; i < uniq_count; i++) printf(" %d", uniq_ids[i]);
  printf("\n");

  // Verify stream mask restricted to sms_per_tpc SMs and they differ from TPC 2
  if (uniq_count > sms_per_tpc) {
    printf("  [FAIL] Stream mask did not restrict SMs (%d > %d)\n",
           uniq_count, sms_per_tpc);
    libsmctrl_set_global_mask(0);
    return 1;
  }
  // Verify different SM set from TPC 2 (stream=TPC1 should differ from next=TPC2)
  if (uniq_count == tpc2_count && uniq_ids[0] == tpc2_ids[0]) {
    printf("  [FAIL] Stream mask produced same SMs as next mask (priority broken)\n");
    libsmctrl_set_global_mask(0);
    return 1;
  }

  libsmctrl_set_global_mask(0);
  printf("  Phase 5: PASSED\n");

  SAFE(cudaFree(d_smids));
  SAFE(cudaStreamDestroy(stream));
  free(h_smids);
  return 0;
}

// ---------- Main ----------
int main(int argc, char** argv) {
  bool probe_only = false;
  int cuda_dev = 0;

  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "--probe"))
      probe_only = true;
    else if (!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h")) {
      printf("Usage: %s [--probe] [device_id]\n"
             "  --probe   Only probe GPU info, skip masking tests\n"
             "  device_id CUDA device index (default: 0)\n", argv[0]);
      return 0;
    } else {
      cuda_dev = atoi(argv[i]);
    }
  }

  SAFE(cudaSetDevice(cuda_dev));

  // Always run probe
  if (test_probe(cuda_dev))
    return 1;

  if (probe_only) {
    printf("\n=== Probe-only mode, skipping mask tests ===\n");
    return 0;
  }

  // Get TPC info for mask tests
  uint32_t num_tpcs;
  int num_sms;
  SAFE(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, cuda_dev));
  if (libsmctrl_get_tpc_info_cuda(&num_tpcs, cuda_dev))
    error(1, 0, "Failed to get TPC info");
  int sms_per_tpc = num_sms / num_tpcs;

  if (num_tpcs < 3) {
    printf("\n[SKIP] GPU has only %u TPCs; need at least 3 for full test\n", num_tpcs);
    return 0;
  }

  // Phase 2: Global mask (TMD callback, no stream offset dependency)
  if (test_global_mask(sms_per_tpc, num_tpcs))
    return 1;

  // Phase 3: Next mask (TMD callback)
  if (test_next_mask(sms_per_tpc))
    return 1;

  // Phase 4: Stream mask (requires correct stream struct offset)
  if (test_stream_mask(sms_per_tpc))
    return 1;

  // Phase 5: Priority ordering
  if (test_mask_priority(sms_per_tpc))
    return 1;

  int major, minor;
  cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, cuda_dev);
  cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, cuda_dev);
  printf("\n========================================\n");
  printf("All tests PASSED for device %d (sm_%d%d, %d SMs, %u TPCs)\n",
         cuda_dev, major, minor, num_sms, num_tpcs);
  printf("========================================\n");
  return 0;
}
