# fork of libsmctrl (you probably want to use the original repo which has new features)
Artifact from [Hardware Compute Partitioning on NVIDIA GPUs*
](https://www.cs.unc.edu/~jbakita/rtas23.pdf) paper. Original repo is [here](http://rtsrv.cs.unc.edu/cgit/cgit.cgi/libsmctrl.git/).

## Supported Platforms

### Mask Types

| Mask Type | CUDA Version | Mechanism |
|-----------|-------------|-----------|
| Global Mask | 6.5 - 13.0 | TMD/QMD Hook |
| Stream Mask | 8.0 - 13.0 | Stream struct |
| Next Mask | 6.5 - 13.0 | TMD/QMD Hook |

### GPU Architectures

| Architecture | Compute Capability | SMs/TPC | TMD Version | Status |
|-------------|-------------------|---------|-------------|--------|
| Kepler V2 | sm_35 - sm_37 | 1 | V01_06 | Supported |
| Maxwell | sm_50 - sm_53 | 1 | V01_06 | Supported |
| Pascal | sm_60 (P100) | 2 | V01_06 | Supported |
| Pascal | sm_61 - sm_62 | 1 | V01_06 | Supported |
| Volta | sm_70 - sm_72 | 2 | V01_06 | Supported |
| Turing | sm_75 | 2 | V01_06 | Supported |
| Ampere | sm_80 - sm_86 | 2 | V01_06 | Supported |
| Ada Lovelace | sm_89 | 2 | V04_00 | Supported |
| Hopper | sm_90 | 2 | V04_00 | Untested |
| Blackwell | sm_120 - sm_121 | 2 | V05_00 | Supported (GB10 verified) |

### Stream Mask Offsets (x86_64)

| CUDA Version | Offset | Tested Driver |
|-------------|--------|---------------|
| 8.0 | 0x0ec | |
| 9.0 - 9.1 | 0x130 | |
| 9.2 | 0x140 | |
| 10.0 - 10.2 | 0x244 | 440.100, 440.82, 440.64, 440.36 |
| 11.0 | 0x274 | |
| 11.1 | 0x2c4 | |
| 11.2 - 11.5 | 0x37c | 470.223.02 |
| 11.6 | 0x38c | |
| 11.7 | 0x3c4 | |
| 11.8 | 0x47c | 520.56.06 |
| 12.0 - 12.1 | 0x4cc | 525.147.05 |
| 12.2 | 0x4e4 | 535.129.03 |
| 12.3 | 0x49c | 545.29.06 |
| 12.4 | 0x4ac | 550.54.14, 550.54.15 |
| 12.5 - 12.6 | 0x4ec | 555.58.02, 560.35.03 |
| 12.7 - 12.8 | 0x4fc | 565.77, 570.124.06 |
| 13.0 | 0x51c | 580.65.06 (validated on x86_64; observed on RTX 4070 Laptop GPU) |

### Stream Mask Offsets (aarch64 / Jetson)

| CUDA Version | Offset | Platform |
|-------------|--------|----------|
| 9.0 | 0x128 | Jetson TX2 (Jetpack 3.x) |
| 10.2 | 0x24c | Jetson AGX Xavier / TX2 (Jetpack 4.x) |
| 11.4 | 0x394 | Jetson AGX Orin (Jetpack 5.x) |
| 12.2 | 0x50c | Jetson AGX Orin (Jetpack 6.x) |
| 12.4 | 0x4c4 | Jetson AGX Orin (Jetpack 6.x + cuda-compat) |
| 12.5 | 0x50c | Jetson AGX Orin (Jetpack 6.x + cuda-compat) |
| 12.6 | 0x514 | Jetson AGX Orin (Jetpack 6.x + cuda-compat) |
| 13.0 | 0x53c | DGX Spark GB10 (driver 580.95.05) |

## Build
```
make libsmctrl.a
```
