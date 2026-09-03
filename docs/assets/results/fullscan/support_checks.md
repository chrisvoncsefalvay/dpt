## FP64 forward and volume backward

Completed resident forward plus volume-only backward uses the full chest resampled to 256³, 360 views, 512² detectors and 1024 midpoint samples. The gradient destination is FP32. Three rounds of ten repetitions run outside Nsight.

| Operation | Median (ms) | p95 (ms) |
|---|---:|---:|
| Forward + volume backward | 9170.87 | 9211.51 |

Volume-gradient relative L2 error against the recorded same-mode reference is 6.83e-08. The adjoint discrepancy is 6.78e-11.

Measured 2026-10-20T03:05:40.691511+00:00 to 2026-10-20T03:10:49.067169+00:00.

The one-view Nsight Compute capture records the following main-kernel counters.

| Main volume kernel counter | Value |
|---|---:|
| DSETP thread instructions | 1,057,720,740 |
| FSETP thread instructions | 3,829,008 |
| REDG thread instructions | 1,399,357,440 |
| Registers per thread | 64 |
| Spill instructions | 0 |
