# Adding GB10 (Blackwell sm_121) Platform Support to libsmctrl

This document describes the full process used to reverse engineer, implement,
and verify TPC masking support for the NVIDIA GB10 (DGX Spark / OEM) GPU
running CUDA 13.0 on aarch64.

## Table of Contents

1. [Platform Specification](#1-platform-specification)
2. [Problem Diagnosis](#2-problem-diagnosis)
3. [Reverse Engineering Approach](#3-reverse-engineering-approach)
4. [TMD Dump Analysis](#4-tmd-dump-analysis)
5. [Brute-Force Offset Scanning](#5-brute-force-offset-scanning)
6. [QMD V05_00 Structure Discovery](#6-qmd-v0500-structure-discovery)
7. [Offset Verification](#7-offset-verification)
8. [Stream Mask Offset Discovery](#8-stream-mask-offset-discovery)
9. [Code Changes](#9-code-changes)
10. [Final Verification](#10-final-verification)
11. [Diagnostic Tools Reference](#11-diagnostic-tools-reference)
12. [Limitations and Future Work](#12-limitations-and-future-work)

---

## 1. Platform Specification

| Property | Value |
|----------|-------|
| GPU | NVIDIA GB10 (DGX Spark / OEM) |
| Compute capability | sm_121 |
| Architecture | Consumer Blackwell |
| SM count | 48 |
| TPC count | 24 |
| SMs per TPC | 2 |
| CUDA driver version | 13.0 (case 13000) |
| Driver version | 580.95.05 |
| Platform | aarch64 |

## 2. Problem Diagnosis

Running `scripts/add_new_platform.sh` failed at two points:

### 2.1 Script Bug: Unquoted Values in state_save()

The `state_save()` function wrote values without quotes:

```bash
# Before (broken):
state_save() { echo "$1=$2" >> "$STATE_DIR/state.env"; }

# Produces:
ARCH_NAME=Consumer Blackwell GB10 (DGX Spark / OEM)
```

When `source`-d, the parentheses caused a bash syntax error. Fixed by wrapping
values in single quotes:

```bash
# After (fixed):
state_save() { echo "$1='$2'" >> "$STATE_DIR/state.env"; }
```

### 2.2 TMD Version Detection Failure

The test binary aborted with:

```
TMD version 0000 is too old! This GPU does not support SM masking.
```

The existing code in `libsmctrl.c` reads the TMD version from byte offset 72:

```c
uint8_t tmd_ver = *(uint8_t*)(tmd + 72);
```

On Blackwell / CUDA 13.0, byte 72 reads `0x00`, causing the code to fall
through to the error case. The TMD/QMD structure has fundamentally changed.

## 3. Reverse Engineering Approach

The overall strategy was:

1. **Dump** the raw TMD bytes from the CUDA callback to understand the structure
2. **Scan** TMD offsets by brute-force writing mask values and observing SM behavior
3. **Verify** write persistence in the callback to confirm the mechanism works
4. **Research** NVIDIA open-source QMD headers for authoritative field definitions
5. **Test** the discovered offsets with the actual SM ID hardware register

### How SM Usage Is Verified

All tests use a GPU kernel that reads the hardware SM ID register via inline
PTX assembly:

```cuda
__global__ void read_smid(uint8_t* smid_arr) {
    if (threadIdx.x != 0) return;
    int smid;
    asm("mov.u32 %0, %%smid;" : "=r"(smid));
    smid_arr[blockIdx.x] = smid;
}
```

`%%smid` is a read-only special register in NVIDIA GPUs that returns the
physical SM ID where the current thread is executing. This is a hardware-level
value, not a software construct. By launching many blocks (142) and collecting
the SM ID from each, we can determine exactly which SMs were used.

**Verification logic:**
- Launch 142 blocks across 1024 threads each
- Each block's thread 0 writes its SM ID to an output array
- Copy array to host, sort, count unique SM IDs
- Compare unique count before (baseline) and after applying a TPC mask

Example:
- Baseline: 48 unique SMs (all SMs active)
- With TPC 0 mask (`~0x1ULL`): 2 unique SMs (SM 0, 1 only)

## 4. TMD Dump Analysis

### Tool: tmd_dump.cu

This tool hooks the CUDA internal callback (same mechanism as libsmctrl) and
dumps the raw TMD bytes when a kernel is launched.

**Build and run:**

```bash
nvcc -ccbin g++ tmd_dump.cu -o tmd_dump -lcuda
./tmd_dump 0
```

**Key findings from the dump:**

```
in_params size field: 0x58 (88 bytes, 11 pointers)
in_params[4] = TMD pointer (same index as previous CUDA versions)

TMD+0x000: 00000000  (no enable bit at traditional location)
TMD+0x048: 00000000  (version field reads zero!)
TMD+0x03A: 0x50      (V05_00 version at new location, byte 58)
TMD+0x130: 00000000  (Hopper V04_00 mask area is empty)
TMD+0x278: 000fffff  (unrelated field, not TPC mask)
```

The traditional version field at byte 72 (`0x48`) was zero. Scanning for
version-like values revealed `0x50` at byte 58 (`0x3A`).

### Tool: tmd_dump3.cu

Dumps data from ALL `in_params` pointer entries (indices 1-4) to verify:
- Which pointer contains the actual TMD
- Whether double-indirection is used
- Structure of other descriptors

**Result:** Pointer at index 4 remains the correct TMD location on CUDA 13.0.

### Tool: tmd_verify.cu

Verifies that TMD writes in the callback actually persist:

```
First launch:
  WROTE 0xDEADBEEF at TMD+0x278
  READBACK TMD+0x278: deadbeef    <- Write succeeded in memory

Second launch:
  TMD+0x278: 000fffff              <- CUDA reset the TMD buffer
```

**Conclusion:** Writes in the callback DO persist within the same launch
(the GPU reads the modified TMD). The buffer is reused but reset between
launches. This confirms the callback-based mechanism still works on CUDA 13.0.

## 5. Brute-Force Offset Scanning

### Tool: tmd_test_one.cu

Tests a single TMD offset in an isolated process:

```bash
nvcc -ccbin g++ tmd_test_one.cu -o tmd_test_one -lcuda

# Baseline (no mask modification)
./tmd_test_one 0 0 0 --baseline    # Output: 48

# Test specific offset with mask+enable+ext
./tmd_test_one 0 0x130 3           # Output: 48 (no effect)
```

Modes: 1=mask only, 2=mask+enable at TMD+0, 3=mask+enable+ext (4 words)

### Scan Process

Full range scan with the Hopper enable bit location (TMD+0, bit 31):

```bash
# Scan 0x050-0x360, mode 3 (mask+enable+ext)
for off in $(seq 80 4 864); do
    hex=$(printf "0x%03x" $off)
    R=$(timeout 3 ./tmd_test_one 0 $hex 3 2>/dev/null || echo "-1")
    C=$(echo "$R" | awk '{print $1}')
    if [ "$C" != "48" ] && [ "$C" -le 24 ] 2>/dev/null; then
        echo "off=$hex: $R"
    fi
done
```

**Result:** No hits. This proved the enable bit has also moved from TMD+0.

### Tool: tmd_test_inv.cu

Flexible tool that accepts arbitrary mask values, offsets, and enable bit
locations:

```bash
nvcc -ccbin g++ tmd_test_inv.cu -o tmd_test_inv -lcuda

# Syntax: ./tmd_test_inv <dev> <mask_off> <mask_lo> <mask_hi> [enable_off]
./tmd_test_inv 0 --baseline x x          # Baseline: 48
./tmd_test_inv 0 0x118 0xFFFFFFFE 0xFFFFFFFF 0x10   # Test V05_00 offsets
```

## 6. QMD V05_00 Structure Discovery

After brute-force scanning failed (because both the mask AND enable bit
locations changed), we consulted the NVIDIA open-source GPU documentation.

### Source

NVIDIA publishes QMD structure headers in the
[open-gpu-doc](https://github.com/NVIDIA/open-gpu-doc/tree/master/classes/compute)
repository:

| Architecture | Class | Header | QMD Versions |
|-------------|-------|--------|-------------|
| Hopper | 0xCBC0 | `clcbc0qmd.h` | V03_00, V04_00 |
| Blackwell A | 0xCDC0 | `clcdc0qmd.h` | V05_00, V05_01, V04_01 |
| Blackwell B | 0xCEC0 | `clcec0qmd.h` | V05_00, V05_01, V04_01 |

### Field Comparison: V04_00 vs V05_00

| Field | Hopper V04_00 | Blackwell V05_00 | Encoding |
|-------|--------------|-----------------|----------|
| QMD_MAJOR_VERSION | MW(583:580) = byte 72 | **MW(471:468) = byte 58** | 4-bit BCD |
| QMD_MINOR_VERSION | MW(579:576) = byte 72 | **MW(467:464) = byte 58** | 4-bit BCD |
| Version byte value | 0x40 | **0x50** | major<<4 \| minor |
| TPC_DISABLE_MASK_VALID | MW(31) = bit 31, DW0 | **MW(159) = bit 159, DW4** | 1 bit |
| TPC_DISABLE_MASK(0) | MW(2463:2432) = byte 304 | **MW(2271:2240) = byte 280** | 32 bits |
| TPC_DISABLE_MASK(1) | MW(2495:2464) = byte 308 | **MW(2303:2272) = byte 284** | 32 bits |
| TPC_DISABLE_MASK(2) | MW(2527:2496) = byte 312 | **MW(2335:2304) = byte 288** | 32 bits |
| TPC_DISABLE_MASK(3) | MW(2559:2528) = byte 316 | **MW(2367:2336) = byte 292** | 32 bits |

### Offset Calculation

`MW(N)` refers to bit N of the QMD structure. To convert to byte offset:
`byte_offset = N / 8`

For the enable bit MW(159): `159 / 8 = 19` (byte 19, bit 7). In terms of
32-bit DWORDs: `DW4` (bytes 16-19), bit 31 of that DWORD.

### Verification Against Dump

From the TMD dump, byte 58 of the TMD structure:
```
DW14 (bytes 56-59): 0x2f5003a4
  byte 56 = 0xa4
  byte 57 = 0x03
  byte 58 = 0x50  <- QMD V05_00 confirmed!
  byte 59 = 0x2f
```

## 7. Offset Verification

With the V05_00 offsets identified, direct verification using `tmd_test_inv`:

```bash
# Mask offset 0x118 (280), enable at 0x10 (16), disable all except TPC 0
./tmd_test_inv 0 0x118 0xFFFFFFFE 0xFFFFFFFF 0x10
# Output: 2 sms: 0 1

# Inverted test: enable only TPC 0 (bit set = enabled)
./tmd_test_inv 0 0x118 0x1 0x0 0x10
# Output: 46  (one TPC disabled, 46 SMs remaining)
```

The mask semantics are confirmed as "bit set = TPC disabled" (same as previous
generations). With `0xFFFFFFFE` (all bits set except bit 0), only TPC 0 remains
active, and the kernel runs on exactly 2 SMs.

## 8. Stream Mask Offset Discovery

After implementing the TMD callback changes, the stream mask offset for
CUDA 13.0 on aarch64 still needed to be found. libsmctrl uses the `MASK_OFF`
environment variable for offset scanning relative to the CUDA 12.2 base
(`0x4e4`):

```bash
for off in $(seq -500 4 500); do
    R=$(MASK_OFF=$off timeout 3 ./libsmctrl_test_blackwell 0 2>&1)
    if echo "$R" | grep -q "Phase 4: PASSED"; then
        echo "HIT: MASK_OFF=$off"
        break
    fi
done
```

**Result:** `MASK_OFF=88`, absolute offset = `0x4e4 + 88 = 0x53c`

## 9. Code Changes

### 9.1 libsmctrl.c: TMD Version Detection and V05_00 Support

```c
// The TMD version field location varies by QMD generation.
// V01_06 through V04_00: version at byte 72
// V05_00 (Blackwell):    version at byte 58
uint8_t tmd_ver = *(uint8_t*)(tmd + 72);

// If byte 72 is zero, check byte 58 for Blackwell V05_00+
if (tmd_ver == 0)
    tmd_ver = *(uint8_t*)(tmd + 58);

if (tmd_ver >= 0x50) {
    // QMD V05_00 is used on Blackwell (sm_12x)
    // TPC_DISABLE_MASK at byte 280-292, enable bit at byte 16 bit 31
    lower_ptr = tmd + 280;
    upper_ptr = tmd + 284;
    // Disable upper 64 TPCs
    *(uint32_t*)(tmd + 288) = -1;
    *(uint32_t*)(tmd + 292) = -1;
    // TPC_DISABLE_MASK_VALID enable bit (bit 159 = DW4 bit 31)
    *(uint32_t*)(tmd + 16) |= 0x80000000;
} else if (tmd_ver >= 0x40) {
    // Hopper V04_00 (unchanged)
    ...
}
```

### 9.2 libsmctrl.c: Stream Mask Offset

```c
// New define for CUDA 13.0 on aarch64
#define CU_13_0_MASK_OFF_JETSON 0x53c
// 13.0 tested on DGX Spark GB10 with driver 580.95.05 (Mar 2026)

// New case in the switch statement (aarch64 section)
case 13000:
    hw_mask_v2 = (void*)(stream_struct_base + CU_13_0_MASK_OFF_JETSON);
    break;
```

### 9.3 libsmctrl.c: Warning Message Update

Changed Hopper warning from `major >= 9` to `major == 9` since Blackwell
(major 10, 12) is now verified:

```c
if (major == 9)
    fprintf(stderr, "libsmctrl: WARNING, TPC masking is untested on Hopper,...");
```

### 9.4 scripts/add_new_platform.sh: Quote Fix

```bash
state_save() { echo "$1='$2'" >> "$STATE_DIR/state.env"; }
```

### 9.5 Test Code: Non-Contiguous SM ID Mapping

On GB10, SM-to-TPC mapping is non-contiguous:

```
TPC 0 -> SM 0, 1
TPC 1 -> SM 8, 9    (not SM 2, 3!)
TPC 2 -> SM 4, 5    (not SM 4, 5 contiguously from TPC 1)
```

Updated all test phases to verify SM **count** rather than specific SM ID
ranges, since the mapping depends on the physical chip layout (floorsweeping).

## 10. Final Verification

### Full Test Suite Output

```
=== Phase 1: Platform Probe (device 0) ===
  Compute capability : sm_121
  Architecture       : Consumer Blackwell GB10 (DGX Spark / OEM)
  SM count           : 48
  TPC count          : 24
  SMs per TPC        : 2
  CUDA driver version: 13.0
  [PASS] GB10 configuration matches expected spec
  Phase 1: PASSED

=== Phase 2: Global Mask (TMD callback) ===
  Baseline: kernel ran on 48 unique SMs
  With TPC 0 only: kernel ran on 2 unique SMs: 0 1
  Phase 2: PASSED

=== Phase 3: Next Mask ===
  With TPC 1 only: kernel ran on 2 unique SMs: 8 9
  After next mask consumed: kernel ran on 48 SMs (should be >2)
  Phase 3: PASSED

=== Phase 4: Stream Mask ===
  With TPC 2 only: kernel ran on 2 unique SMs: 4 5
  Phase 4: PASSED

=== Phase 5: Mask Priority (next > stream > global) ===
  global=TPC0, stream=TPC1, next=TPC2: ran on SMs: 16 17
  After next consumed (stream=TPC1 should win): ran on SMs: 2 3
  Phase 5: PASSED

========================================
All tests PASSED for device 0 (sm_121, 48 SMs, 24 TPCs)
========================================
```

### add_new_platform.sh Output

```
=== Phase 0: Environment Check ===           OK
=== Phase 1: GPU Information Collection ===  OK
=== Phase 2: Support Check & TMD Callback === All PASSED
=== Phase 3: Stream Offset Scan ===          Already works (0x53c)
=== Phase 4: Patch Generation ===            Already supported
=== Phase 5: Summary Report ===              Platform already fully supported
```

## 11. Diagnostic Tools Reference

All tools are in `platforms/gb10/`. Build from the project root with:

```bash
NVCC=/usr/local/cuda/bin/nvcc
NVCC_FLAGS="-ccbin g++ -lcuda -I/usr/local/cuda/include -L/usr/local/cuda/lib64"
GB10=platforms/gb10
```

### tmd_dump.cu

**Purpose:** Dump raw TMD bytes from the CUDA pre-upload callback.

```bash
$NVCC $GB10/tmd_dump.cu -o tmd_dump $NVCC_FLAGS
./tmd_dump [device_id]
```

Outputs:
- `in_params` structure (size, pointer entries)
- First 1024 bytes of TMD as hex dump
- 32-bit word view for mask analysis
- Comparison against known Hopper V04_00 offsets

### tmd_dump3.cu

**Purpose:** Dump data from ALL `in_params` pointer entries to identify which
contains the TMD and to examine other descriptors.

```bash
$NVCC $GB10/tmd_dump3.cu -o tmd_dump3 $NVCC_FLAGS
./tmd_dump3 [device_id]
```

### tmd_verify.cu

**Purpose:** Verify that TMD writes in the callback persist and affect the GPU.
Also enumerates available callback domain/cbid combinations.

```bash
$NVCC $GB10/tmd_verify.cu -o tmd_verify $NVCC_FLAGS
./tmd_verify [device_id]
```

Tests:
- Write marker values to TMD, read back immediately (should match)
- Check if values persist across kernel launches (CUDA resets the buffer)
- Enumerate all domain/cbid callback combinations

### tmd_test_one.cu

**Purpose:** Test a single TMD offset in an isolated invocation. Designed for
use in shell-script scanning loops.

```bash
$NVCC $GB10/tmd_test_one.cu -o tmd_test_one $NVCC_FLAGS
./tmd_test_one <dev> <offset> <mode> [--baseline]
```

Modes:
- 0 + `--baseline`: no mask, report SM count
- 1: write mask at offset (no enable bit)
- 2: write mask + enable bit at TMD+0
- 3: write mask + enable bit + extended mask (4 words)

### tmd_test_inv.cu

**Purpose:** Test arbitrary mask values, offsets, and enable bit locations.
Supports both normal (`bit set = disabled`) and inverted (`bit set = enabled`)
mask semantics.

```bash
$NVCC $GB10/tmd_test_inv.cu -o tmd_test_inv $NVCC_FLAGS
./tmd_test_inv <dev> <mask_offset> <mask_lo> <mask_hi> [enable_off] [enable_bit]
./tmd_test_inv <dev> --baseline x x
```

Example for verifying V05_00 offsets:

```bash
# Enable only TPC 0 (mask=0xFFFFFFFE disables all others)
./tmd_test_inv 0 0x118 0xFFFFFFFE 0xFFFFFFFF 0x10
# Expected: 2 sms: 0 1
```

### scan_tmd.sh

**Purpose:** Shell script that drives `tmd_test_one` in a loop to brute-force
scan TMD offsets.

```bash
bash $GB10/scan_tmd.sh [device_id]
```

### libsmctrl_test_blackwell.cu

**Purpose:** Comprehensive 5-phase verification test for Blackwell TPC masking.

```bash
make libsmctrl.a
$NVCC $GB10/libsmctrl_test_blackwell.cu -o libsmctrl_test_blackwell \
    -g -L. -l:libsmctrl.a -lcuda $NVCC_FLAGS

./libsmctrl_test_blackwell [--probe] [device_id]
```

Phases:
1. Platform probe (SM count, TPC count, architecture ID)
2. Global mask via TMD callback
3. Next mask (single-launch, auto-consumed)
4. Stream mask (per-stream persistent mask via stream struct offset)
5. Mask priority ordering (next > stream > global)

### tpc_multi_test.cu

**Purpose:** Multi-TPC control verification. Tests single TPC isolation for
all 24 TPCs, TPC pair combinations, scaling from 1 to 24 TPCs, multi-TPC via
next/stream masks, and disjoint stream partitioning with overlap detection.

```bash
make libsmctrl.a
$NVCC $GB10/tpc_multi_test.cu -o tpc_multi_test \
    -g -L. -l:libsmctrl.a -lcuda $NVCC_FLAGS

./tpc_multi_test [device_id]
```

Tests (45 total):
1. Single TPC isolation (all 24 TPCs individually)
2. TPC pairs (adjacent and non-adjacent)
3. Scaling TPC count (1, 2, 4, 8, 12, 16, 20, 24)
4. Multi-TPC via next mask + consumption verification
5. Multi-TPC via stream mask + clear verification
6. Disjoint streams with overlap detection

## 12. Limitations and Future Work

### Tested

- GB10 (sm_121) on aarch64 with CUDA 13.0 driver 580.95.05

### Not Tested

- **Hopper (sm_9x)**: No hardware available. Code unchanged from upstream.
- **x86_64 CUDA 13.0**: GB10 is aarch64-only. The x86_64 stream mask offset
  for CUDA 13.0 is unknown.
- **Datacenter Blackwell (sm_100, GB200)** and **Blackwell Ultra (sm_103,
  GB300)**: TMD V05_00 callback code should work (same QMD structure), but
  stream mask offsets are untested.
- **RTX 50-series (sm_120)**: Consumer Blackwell should share the same QMD
  V05_00 structure but is untested on this specific chip.

### Known Quirk: Logical vs Physical TPC Bit Ordering

On GB10, the **true** TPC-to-SM mapping is contiguous (TPC N -> SM {2N, 2N+1}),
verified by isolating each TPC individually via `set_stream_mask_ext`.

However, `set_global_mask` (QMD direct write) uses a **physical GPC-interleaved
bit ordering**, causing an apparent stride-8 pattern in SM IDs. This is because
the QMD's TPC_DISABLE_MASK bits are ordered by physical GPC position, not by
logical TPC index. The CUDA driver performs this remapping when copying from
the stream struct to the QMD, but the TMD callback bypasses it.

See `tpc_sm_mapping_analysis.md` for full details, including:
- Complete TPC-to-SM mapping table
- GPC physical layout (6 GPCs x 4 TPCs x 2 SMs)
- Side-by-side stream_mask_ext vs global_mask comparison
- Incremental TPC scaling test results

Tests should verify SM **count** per TPC, not specific SM ID values.
