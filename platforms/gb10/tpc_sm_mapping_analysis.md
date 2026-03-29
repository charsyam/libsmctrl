# GB10 TPC-SM Mapping Analysis

Date: 2026-03-29
Platform: GB10 (sm_121), CUDA 13.0, Driver 580.95.05, aarch64

## Summary

GB10 has 48 SMs, 24 TPCs (2 SMs/TPC), organized in 6 GPCs (4 TPCs/GPC).

The TPC-to-SM mapping is **contiguous** when accessed via the logical TPC index
(stream mask). However, the QMD (TMD callback) uses a **physical GPC-interleaved
bit ordering**, causing an apparent stride-8 pattern in SM IDs.

## True TPC-to-SM Mapping (Logical Order)

Verified by isolating each TPC individually via `libsmctrl_set_stream_mask_ext`:

```
TPC  0 -> SM  0,  1      TPC 12 -> SM 24, 25
TPC  1 -> SM  2,  3      TPC 13 -> SM 26, 27
TPC  2 -> SM  4,  5      TPC 14 -> SM 28, 29
TPC  3 -> SM  6,  7      TPC 15 -> SM 30, 31
TPC  4 -> SM  8,  9      TPC 16 -> SM 32, 33
TPC  5 -> SM 10, 11      TPC 17 -> SM 34, 35
TPC  6 -> SM 12, 13      TPC 18 -> SM 36, 37
TPC  7 -> SM 14, 15      TPC 19 -> SM 38, 39
TPC  8 -> SM 16, 17      TPC 20 -> SM 40, 41
TPC  9 -> SM 18, 19      TPC 21 -> SM 42, 43
TPC 10 -> SM 20, 21      TPC 22 -> SM 44, 45
TPC 11 -> SM 22, 23      TPC 23 -> SM 46, 47
```

Formula: `TPC N -> SM {2N, 2N+1}`

## GPC Physical Layout

6 GPCs, each with 8 SM slots (4 TPCs x 2 SMs/TPC):

```
GPC 0: SM  0- 7  (TPC  0, 1, 2, 3)
GPC 1: SM  8-15  (TPC  4, 5, 6, 7)
GPC 2: SM 16-23  (TPC  8, 9,10,11)
GPC 3: SM 24-31  (TPC 12,13,14,15)
GPC 4: SM 32-39  (TPC 16,17,18,19)
GPC 5: SM 40-47  (TPC 20,21,22,23)
```

## stream_mask_ext vs global_mask Behavior Difference

When enabling TPCs 0-3 simultaneously:

| API                      | TPC bits set | SM IDs observed        | Pattern    |
|--------------------------|-------------|------------------------|------------|
| `set_stream_mask_ext`    | 0-3         | 0,1,2,3,4,5,6,7       | Contiguous |
| `set_global_mask`        | 0-3         | 0,1,8,9,16,17,24,25   | Stride 8   |

### Root Cause

The two APIs write TPC masks at different stages, and the bit ordering differs:

1. **`set_stream_mask_ext`** writes to the CUDA stream struct. CUDA's driver
   copies this mask into the QMD, applying a logical-to-physical TPC remapping.
   Bit N in the stream struct corresponds to logical TPC N.

2. **`set_global_mask`** writes directly to the QMD in the pre-upload callback.
   Bit N in the QMD's TPC_DISABLE_MASK corresponds to a **physical** TPC index,
   which is GPC-interleaved:
   - QMD bit 0 -> GPC 0, TPC 0 (logical TPC 0)
   - QMD bit 1 -> GPC 1, TPC 0 (logical TPC 4)
   - QMD bit 2 -> GPC 2, TPC 0 (logical TPC 8)
   - QMD bit 3 -> GPC 3, TPC 0 (logical TPC 12)
   - QMD bit 4 -> GPC 4, TPC 0 (logical TPC 16)
   - QMD bit 5 -> GPC 5, TPC 0 (logical TPC 20)
   - QMD bit 6 -> GPC 0, TPC 1 (logical TPC 1)
   - ...

   So enabling QMD bits 0-3 selects one TPC from each of GPC 0-3, giving
   SMs {0,1}, {8,9}, {16,17}, {24,25} (stride 8 = SM slots per GPC).

### Implication

The stride-8 pattern is **not a bug** -- it reflects the QMD's physical TPC
bit ordering vs the stream struct's logical ordering. Both APIs correctly
constrain the kernel to the expected number of SMs (2 per TPC).

For users who need specific logical TPC control, `set_stream_mask_ext` provides
the expected sequential behavior. `set_global_mask` works correctly for SM count
control but the TPC-to-GPC assignment follows the hardware's physical layout.

## Incremental TPC Test Results

Verified TPC 1-4 (incremental) with `set_global_mask`:

```
TPC 0..0 enabled (1 TPC):  2 SMs -> SM 0, 1                          PASS
TPC 0..1 enabled (2 TPCs): 4 SMs -> SM 0, 1, 8, 9                    PASS
TPC 0..2 enabled (3 TPCs): 6 SMs -> SM 0, 1, 8, 9, 16, 17            PASS
TPC 0..3 enabled (4 TPCs): 8 SMs -> SM 0, 1, 8, 9, 16, 17, 24, 25    PASS
```

Verified TPC 1-8 with `set_stream_mask_ext`:

```
TPC 0..0 (1 TPC):  2 SMs -> SM 0, 1                                   PASS
TPC 0..1 (2 TPCs): 4 SMs -> SM 0, 1, 2, 3                             PASS
TPC 0..2 (3 TPCs): 6 SMs -> SM 0, 1, 2, 3, 4, 5                       PASS
TPC 0..3 (4 TPCs): 8 SMs -> SM 0, 1, 2, 3, 4, 5, 6, 7                 PASS
TPC 0..4 (5 TPCs):10 SMs -> SM 0, 1, 2, 3, 4, 5, 6, 7, 8, 9           PASS
TPC 0..5 (6 TPCs):12 SMs -> SM 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11  PASS
TPC 0..6 (7 TPCs):14 SMs -> SM 0-13                                    PASS
TPC 0..7 (8 TPCs):16 SMs -> SM 0-15                                    PASS
```

## Not Related: Grace Blackwell NUMA Index Pattern

Some Grace Blackwell systems (e.g., GB200 NVL72) show NUMA indices incrementing
by 8. This is a **system-level memory topology** feature (multi-die/NVLink), not
related to the intra-GPU GPC/TPC layout described here.

| Pattern          | Cause                              | Level              |
|------------------|------------------------------------|--------------------|
| SM stride 8      | GPC physical layout (6 GPCs x 8 SM slots) | Single GPU chip |
| NUMA index x8    | Multi-die memory controller/NVLink topology | System-wide    |

The matching stride value (8) is coincidental.

## Test Tools

All test source files in `platforms/gb10/`:

| File | Purpose |
|------|---------|
| `libsmctrl_test_tpc_sm_map.cu`     | Maps each TPC to its SM IDs individually |
| `libsmctrl_test_tpc_incremental.cu`| Tests TPC 0..N scaling with global_mask |
| `libsmctrl_test_tpc_multi.cu`      | Compares stream_mask_ext vs global_mask |

### Build

```bash
NVCC=/usr/local/cuda/bin/nvcc
FLAGS="-ccbin g++ -g -L. -l:libsmctrl.a -lcuda -I/usr/local/cuda/include -L/usr/local/cuda/lib64"
GB10=platforms/gb10

make libsmctrl.a
$NVCC $GB10/libsmctrl_test_tpc_sm_map.cu      -o libsmctrl_test_tpc_sm_map      $FLAGS
$NVCC $GB10/libsmctrl_test_tpc_incremental.cu  -o libsmctrl_test_tpc_incremental $FLAGS
$NVCC $GB10/libsmctrl_test_tpc_multi.cu        -o libsmctrl_test_tpc_multi       $FLAGS
```
