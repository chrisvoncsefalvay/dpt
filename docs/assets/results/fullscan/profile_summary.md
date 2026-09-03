# Fullscan profiling evidence

Nsight Compute durations are replay-profiled kernel sums. Nsight Systems ranges are instrumented intervals. Completed-call medians are reported in the timing tables.

FLOPs count predicated-thread ADD + MUL + 2 * FMA. Missing counters are marked unavailable.

## Nsight Compute capture scope

- **fp32-volume**: full-extent resampled chest: 256x256x256, 1 zero-angle view (saved index 0), 512x512, 1024 midpoints. volume VJP. complete probe_backward NVTX operation. Completed `2026-10-20T04:03:21.456530+00:00`.
- **fp32-joint**: full-extent resampled chest: 256x256x256, 1 zero-angle view (saved index 0), 512x512, 1024 midpoints. joint VJP. complete probe_backward NVTX operation. Completed `2026-10-20T04:04:53.330725+00:00`.
- **fp64-volume**: full-extent resampled chest: 256x256x256, 1 zero-angle view (saved index 0), 512x512, 1024 midpoints. volume VJP. complete probe_backward NVTX operation. Completed `2026-10-20T04:01:27.164581+00:00`.
- **fp64-joint**: full-extent resampled chest: 256x256x256, 1 zero-angle view (saved index 0), 512x512, 1024 midpoints. joint VJP. complete probe_backward NVTX operation. Completed `2026-10-20T04:02:59.953929+00:00`.

Roofline figures use the named projector/VJP kernels and theoretical NVIDIA ceilings at their recorded clocks. The numeric report contains every captured kernel, input/source digest, command and report hash.

| Capture | Kernels | Replay kernel sum (ms) | FP32 FLOPs | FP64 FLOPs |
|---|---:|---:|---:|---:|
| fp32-volume | 6 | 1.165 | 7,716,865,840 | 25,952,309 |
| fp32-joint | 56 | 2.208 | 26,805,386,356 | 55,735,104 |
| fp64-volume | 6 | 11.866 | 3,829,008 | 5,310,689,589 |
| fp64-joint | 56 | 30.922 | 5,543,760 | 21,281,483,236 |

| Projector/VJP kernel | Registers/thread | Occupancy (%) | FP64 pipe active (%) | Spills | Local sector bytes |
|---|---:|---:|---:|---:|---:|
| fp32-volume:k4 | 64 | 40.89 | 2.03 | 0 | 0 |
| fp32-joint:k5 | 128 | 30.82 | 2.66 | 0 | 0 |
| fp64-volume:k4 | 64 | 45.01 | 84.86 | 0 | 0 |
| fp64-joint:k5 | 150 | 23.67 | 84.38 | 0 | 0 |

The rooflines use outgoing L1/L2 writeback traffic. The table below reports total L2 requested-sector bytes, `lts__t_bytes.sum = 32 * lts__t_sectors.sum`, including atomic and reduction requests. Each intensity uses the byte count in its column.

| Projector/VJP kernel | Total L2 requested-sector bytes | FP32 FLOP/L2 requested-sector byte | FP64 FLOP/L2 requested-sector byte |
|---|---:|---:|---:|
| fp32-volume:k4 | 1,869,343,104 | 4.128116 | 0.013883 |
| fp32-joint:k5 | 4,074,682,016 | 6.578522 | 0.013643 |
| fp64-volume:k4 | 20,400,672,320 | 0.000188 | 0.260319 |
| fp64-joint:k5 | 20,788,120,032 | 0.000267 | 1.023726 |

## Nsight Systems application ranges

NVTX intervals include CPU execution and waits. Kernel sums report GPU execution. These intervals overlap. Trace gaps measure intervals with no attributed GPU activity.

| Capture | Precision | Gradients | Views | Range | NVTX (ms) | Kernels | Kernel sum (ms) | Volume scans | Scan time (ms) | 4-byte D2H | Sync APIs | Trace gaps (ms) |
|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fp32-volume-timeline | fp32 | volume | 360 | probe_forward #1 | 509.686 | 363 | 505.105 | 1 | 0.048 | 0 | 1 | 4.299 |
| fp32-volume-timeline | fp32 | volume | 360 | probe_backward #1 | 544.924 | 1084 | 539.431 | 2 | 0.063 | 0 | 1 | 5.212 |
| fp32-joint-timeline | fp32 | volume+pose | 360 | probe_forward #1 | 513.653 | 385 | 507.910 | 1 | 0.050 | 0 | 3 | 5.463 |
| fp32-joint-timeline | fp32 | volume+pose | 360 | probe_backward #1 | 961.167 | 2238 | 952.174 | 2 | 0.063 | 0 | 7 | 8.705 |
| fp64-volume-timeline | fp64 | volume | 360 | probe_forward #1 | 4,926.743 | 363 | 4,922.231 | 1 | 0.049 | 0 | 1 | 4.231 |
| fp64-volume-timeline | fp64 | volume | 360 | probe_backward #1 | 4,308.271 | 1084 | 4,302.482 | 2 | 0.062 | 0 | 1 | 5.506 |
| fp64-joint-timeline | fp64 | volume+pose | 360 | probe_forward #1 | 4,929.001 | 385 | 4,923.960 | 1 | 0.050 | 0 | 3 | 4.756 |
| fp64-joint-timeline | fp64 | volume+pose | 360 | probe_backward #1 | 11,993.697 | 2238 | 11,985.570 | 2 | 0.062 | 0 | 7 | 7.841 |

`fp32-volume-timeline`: volume derivative only. Fixed identity geometry without a pose tensor. Seed: `data/chest-256-cotangent.npy`. Completed `2026-10-20T04:07:22.030598+00:00`.

`fp32-joint-timeline`: both field/pose active. Identity pose. Seed: `data/chest-256-cotangent.npy`. Completed `2026-10-20T04:07:36.574255+00:00`.

`fp64-volume-timeline`: volume derivative only. Fixed identity geometry without a pose tensor. Seed: `data/chest-256-cotangent.npy`. Completed `2026-10-20T04:06:08.708776+00:00`.

`fp64-joint-timeline`: both field/pose active. Identity pose. Seed: `data/chest-256-cotangent.npy`. Completed `2026-10-20T04:07:09.714409+00:00`.
