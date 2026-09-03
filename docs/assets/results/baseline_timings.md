| Existing mode | Resident forward median / p95 (ms) | Checked public call median / p95 (ms) | Completed NumPy call median / p95 (ms) | Image RMSE vs strict |
|---|---:|---:|---:|---:|
| Strict | 23.790 / 24.129 | 24.274 / 24.632 | 24.824 / 25.216 | 0 |
| Mixed FP32 | 12.825 / 13.027 | 13.309 / 13.435 | 13.866 / 14.004 | 2.07858e-07 |
| Fast FP32 | 4.946 / 4.987 | 5.260 / 5.310 | 5.788 / 5.873 | 2.07876e-07 |

GPU: NVIDIA RTX PRO 6000 Blackwell Workstation Edition. Native CT: 512 × 512 × 139 voxels. Three views, each 512 × 512 pixels. 1024 samples per ray, clipped midpoint integration.

Resident timings use device events. Checked public and completed NumPy timings use synchronised wall time. All entries include the three views. These existing modes use FP32 field storage and FP64 ray accumulation/output.

Actual measurement interval: 2026-10-19T14:14:17.698740+00:00 to 2026-10-19T14:15:04.343468+00:00.
