# Platform-Specific Test & Diagnostic Tools

Each subdirectory contains the diagnostic tools, test code, and documentation
used to reverse engineer and verify TPC masking support for a specific GPU
platform.

## Directory Structure

```
platforms/
  README.md              <- this file
  gb10/                  <- NVIDIA GB10 (DGX Spark, sm_121, CUDA 13.0, aarch64)
    add_gb10_platform.md <- full write-up of the reverse engineering process
    tpc_sm_mapping_analysis.md <- TPC-SM mapping & logical vs physical bit ordering
    libsmctrl_test_blackwell.cu  <- 5-phase verification test
    tpc_multi_test.cu    <- multi-TPC control test (45 tests)
    libsmctrl_test_tpc_sm_map.cu      <- per-TPC SM ID mapper
    libsmctrl_test_tpc_incremental.cu <- incremental TPC 1-4 scaling test
    libsmctrl_test_tpc_multi.cu       <- stream_mask_ext vs global_mask comparison
    tmd_dump.cu          <- TMD raw byte dumper
    tmd_dump2.cu         <- extended TMD dumper (enable bit scanning)
    tmd_dump3.cu         <- all in_params pointer dumper
    tmd_scan.cu          <- brute-force TMD offset scanner (single-process)
    tmd_scan2.cu         <- TMD scanner with fork-based isolation (failed on CUDA)
    tmd_test_one.cu      <- single-offset tester (for shell-script loops)
    tmd_test_inv.cu      <- flexible tester (arbitrary mask/offset/enable)
    tmd_verify.cu        <- TMD write persistence verifier
    scan_tmd.sh          <- shell driver for tmd_test_one brute-force scan
```

## Adding a New Platform

When adding support for a new GPU architecture or CUDA version:

1. **Create a new directory** under `platforms/` named after the GPU
   (e.g., `platforms/gb300/`, `platforms/rtx5090/`)

2. **Copy and adapt the diagnostic tools** from an existing platform directory.
   The GB10 tools are designed to be reusable -- the key tools to start with:
   - `tmd_dump.cu` -- dump raw TMD to find version field and structure layout
   - `tmd_test_inv.cu` -- test candidate offsets with arbitrary mask/enable values
   - `scan_tmd.sh` + `tmd_test_one.cu` -- brute-force offset scanning

3. **Follow the reverse engineering process** documented in the platform's
   markdown file. The general steps are:
   - Run `tmd_dump` to get the raw TMD structure
   - Check NVIDIA's [open-gpu-doc](https://github.com/NVIDIA/open-gpu-doc/tree/master/classes/compute)
     for QMD header files matching the new architecture
   - Identify version field location, mask offsets, and enable bit
   - Verify with `tmd_test_inv`
   - Scan for stream mask offset using `MASK_OFF` env var
   - Run full test suite

4. **Write a documentation file** (e.g., `add_<platform>.md`) covering:
   - Platform specification (SM count, TPC count, compute capability)
   - QMD version and field offsets discovered
   - Changes made to `libsmctrl.c`
   - Full test output
   - Any platform-specific quirks (e.g., non-contiguous SM mapping)

## Building Tools

From the project root:

```bash
NVCC=/usr/local/cuda/bin/nvcc
NVCC_FLAGS="-ccbin g++ -lcuda -I/usr/local/cuda/include -L/usr/local/cuda/lib64"
PLATFORM=platforms/gb10  # change as needed

# Standalone tools (no libsmctrl dependency)
$NVCC $PLATFORM/tmd_dump.cu    -o tmd_dump    $NVCC_FLAGS
$NVCC $PLATFORM/tmd_test_inv.cu -o tmd_test_inv $NVCC_FLAGS

# Tests that link against libsmctrl
make libsmctrl.a
$NVCC $PLATFORM/libsmctrl_test_blackwell.cu -o libsmctrl_test_blackwell \
    -g -L. -l:libsmctrl.a -lcuda $NVCC_FLAGS
$NVCC $PLATFORM/tpc_multi_test.cu -o tpc_multi_test \
    -g -L. -l:libsmctrl.a -lcuda $NVCC_FLAGS
```
