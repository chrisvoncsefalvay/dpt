# Full-scan volume-only iterations

GPU: NVIDIA RTX PRO 6000 Blackwell Workstation Edition. Complete chest resampled to 256³, with 360 views at 512².

Each completed iteration requests volume gradients only. DPT uses a fresh forward graph and the exact realised sampling transpose. Inputs stay on CUDA.

| Implementation | Algorithm / sampling | Boundary | Wall median ms | Wall p95 ms | Event median ms | Event p95 ms | Sampled process GiB |
|---|---|---|---|---|---|---|---|
| DPT | FP32 midpoint, 1024 | CUDA resident | 1058 | 1060.35 | 1058.01 | 1060.36 | 1.61133 |
| astra-ss1 | ASTRA cuda3d native forward and voxel-driven native backprojection | benchmark autograd adapter around resident direct_FP/direct_BP | 49.8062 | 50.655 | unavailable | unavailable | 2.86133 |
| tigre-siddon | TIGRE Siddon forward. explicit matched native Atb | shipped TIGRE Torch A forward. adapter uses explicit matched Atb backward | 481.821 | 507.618 | unavailable | unavailable | 1.80859 |

Wall timings cover completed calls. Device events use the owning CUDA stream where recorded. NVML memory is the largest sampled process allocation.

## Numerical checks

| Run | Adjoint relative discrepancy | Volume VJP relative L2 vs same-sample reference |
|---|---|---|
| DPT N1024 | 3.46252e-10 | 4.62702e-08 |

## Alternative algorithms

| Tested adapter | Image relative L2 vs DPT FP64 N1024 | BP relative L2 vs DPT FP64 N1024 VJP | Measured adjoint defect |
|---|---|---|---|
| astra-ss1 | 0.000826789 | 0.0438604 | 0.0272748 |
| tigre-siddon | 0.00314858 | 0.00591296 | 8.69585e-05 |
| tigre-interpolated-1 | 0.000613616 | unavailable | unavailable |
| tigre-interpolated-0.5 | 0.000493744 | unavailable | unavailable |
| tigre-interpolated-0.25 | 0.000482204 | unavailable | unavailable |

Image and backprojection differences use the recorded DPT FP64 1024-midpoint calculation. The adjoint defect tests each backend's own forward/backprojection pair on the saved field and cotangent.

ASTRA uses CUDA DLPack resident buffers in its Torch adapter. The TIGRE Torch adapter includes host copies and explicit matched Atb.

## Component timing boundaries

| Measured component | Completed boundary | Wall median ms | Wall p95 ms |
|---|---|---|---|
| astra-ss1/resident/forward | preallocated CUDA DLPack direct_FP/direct_BP. zeros and device synchronisation included | 38.0995 | 38.4807 |
| astra-ss1/resident/backprojection | preallocated CUDA DLPack direct_FP/direct_BP. zeros and device synchronisation included | 9.07474 | 9.52741 |
| astra-ss1/torch/forward | benchmark autograd adapter around resident direct_FP/direct_BP | 38.3124 | 38.8757 |
| astra-ss1/torch/backprojection | benchmark autograd adapter around resident direct_FP/direct_BP | 9.09839 | 9.23731 |
| astra-ss1/host/forward | native CPU DLPack direct_FP/direct_BP. ASTRA performs required transfers. completed host arrays returned | 94.9852 | 98.3557 |
| astra-ss1/host/backprojection | native CPU DLPack direct_FP/direct_BP. ASTRA performs required transfers. completed host arrays returned | 98.7512 | 111.828 |
| tigre-siddon/torch/forward | shipped TIGRE Torch A forward. adapter uses explicit matched Atb backward | 227.626 | 232.496 |
| tigre-siddon/torch/backprojection | shipped TIGRE Torch A forward. adapter uses explicit matched Atb backward | 239.297 | 254.077 |
| tigre-siddon/host/forward | native Ax/Atb NumPy inputs and outputs. required GPU transfers/allocation included | 177.071 | 182.561 |
| tigre-siddon/host/backprojection | native Ax/Atb NumPy inputs and outputs. required GPU transfers/allocation included | 67.8292 | 71.6855 |
| tigre-interpolated-1/host/forward | native Ax/Atb NumPy inputs and outputs. required GPU transfers/allocation included | 149.739 | 163.984 |
| tigre-interpolated-0.5/host/forward | native Ax/Atb NumPy inputs and outputs. required GPU transfers/allocation included | 152.872 | 160.877 |
| tigre-interpolated-0.25/host/forward | native Ax/Atb NumPy inputs and outputs. required GPU transfers/allocation included | 285.955 | 288.462 |

## Measurement identity

| Run | Completed UTC | Package source-map SHA256 |
|---|---|---|
| DPT N1024 | 2026-10-20T03:56:16.237333+00:00 | 01643d02a029127f9a1ac3c04a96b6dda52e91fe5feade622e773fe698e98749 |
| astra-ss1 | 2026-10-19T19:08:46.183146+00:00 | astra-ss1: 878118cff04ce5c680a93f32acdbcd75b59d2228fdee70bbfdd7be8e8cbe2bd4 |
| tigre-siddon | 2026-10-19T19:55:09.966884+00:00 | tigre-siddon: 8a9be206ab714b9ff06e6eb82f068e30e8afe5a01ceb952eca141eb19b42df41 |
| tigre-interpolated-1, tigre-interpolated-0.5, tigre-interpolated-0.25 | 2026-10-19T19:11:16.627127+00:00 | tigre-interpolated-1: 8a9be206ab714b9ff06e6eb82f068e30e8afe5a01ceb952eca141eb19b42df41, tigre-interpolated-0.5: 8a9be206ab714b9ff06e6eb82f068e30e8afe5a01ceb952eca141eb19b42df41, tigre-interpolated-0.25: 8a9be206ab714b9ff06e6eb82f068e30e8afe5a01ceb952eca141eb19b42df41 |

Raw samples include round dates and device context. Competitor identities use the recorded backend source files and versions.
