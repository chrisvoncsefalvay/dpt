# Preparing a case

The application drivers read images, volumes and calibration from a JSON case manifest. It lists the input files, their units and how they were prepared. File hashes identify the exact input bytes.

A case is UTF-8 JSON with integer `schema_version: 1`, a non-empty `arrays` object and an application section named `registration`, `reconstruction` or `acquisition_design`. Repeated JSON keys are rejected. Application fields refer to names in `arrays`.

## Declare arrays explicitly

Every named array requires these fields:

| Field | Contract |
|---|---|
| `path` | Existing `.npy` file, absolute or relative to the case JSON |
| `sha256` | Actual SHA-256 file digest, 64 hexadecimal characters |
| `units` | Exact unit string required by the application role |
| `source` | Non-empty source identity and preparation/calibration description |
| `rights` | Non-empty description of applicable use and redistribution terms |

Array names begin with a letter or digit, contain at most 128 characters and use letters, digits, dots, underscores or hyphens. Files must contain one native-endian, C-contiguous NumPy array with the required dtype and shape. Loading preserves its dtype and shape. Object arrays, archives, non-finite values and symlink files are rejected.

Compute the digest after saving the final prepared file. The loader checks it against the declared digest and again against the bytes read for array loading. Keep the case and its files unchanged throughout a run. The recorder snapshots every declared array, including optional evaluation references, and rejects successful completion if a declared source changes.

The source description records conversions performed before loading. Convert Hounsfield units to attenuation in `mm^-1` before supplying the array. Record the conversion method, calibration and any mask preprocessing, retaining the original source as required by the study's provenance and rights.

## Geometry and array order

Grid JSON contains `shape: [nz, ny, nx]`, `spacing_mm: [hx, hy, hz]`, `origin_mm: [x, y, z]` and an optional nine-entry row-major `orientation`. Volume arrays have shape `(nz, ny, nx)`, with x contiguous. Grid origin is the first sample centre.

Detector JSON contains `source_mm`, `origin_mm`, `u`, `v`, `spacing_mm: [column, row]` and `shape: [height, width]`. Images have shape `(height, width)`. Detector origin is the first pixel centre. A rigid transform uses nine row-major `rotation` entries and three `translation_mm` entries, mapping object coordinates into world coordinates. See [geometry conventions](../concepts/geometry.md) before preparing a case.

## Registration

The `registration` section requires `grid`, `attenuation`, `chart`, `samples_per_ray`, `policy` and a non-empty `views` list. Attenuation is non-negative FP32 with grid shape and units `mm^-1`. The chart contains an `anchor` transform and six positive `scales`, ordered translation x/y/z then rotation x/y/z, in mm and radians. It may declare `rotation_radius_radians`. `policy` supplies `RecoveryPolicy` keyword settings.

Each view requires unique `id`, `geometry`, `observation`, `mask` and positive scalar `open_beam_counts`. IDs use letters, digits, underscores and hyphens. Observations are FP32 integer counts in `[0, 2**24]`, with units `counts`. Masks are FP32 binary images labelled `dimensionless`, with at least one included pixel. Excluded observations must already be zero. Record that preprocessing. Precision defaults to `float64` and integration to `midpoint`. `cell_gauss` is also accepted.

Optional `evaluation` requires `reference_pose`, FP64 object-frame `landmarks` of shape `(N,3)` in `mm`, and non-empty `source` and `uncertainty` descriptions. The driver reads these for scoring after fitting fixes the accepted pose.

## Reconstruction

The `reconstruction` section requires `grid`, `fixed_pose`, `initial_volume`, `samples_per_ray`, `views` and `solver`. Initial attenuation is non-negative FP32 in `mm^-1`. Each view supplies `geometry`, `counts` and `open_beam` array names. Both images are FP32 in `counts`. Observations are integers in `[0, 2**24]`, whilst open-beam means are strictly positive and need not be integers.

All seven solver controls are required:

| Control | Domain |
|---|---|
| `iterations`, `maximum_backtracks` | Positive integers |
| `initial_step_mm_inverse_squared` | Positive, in `mm^-2` |
| `backtracking_factor`, `armijo` | Strictly between zero and one |
| `gradient_mapping_tolerance_mm` | Non-negative, in `mm` |
| `regularisation_mm` | Non-negative, in `mm` |

Optional positive `mapping_step_mm_inverse_squared` fixes the stationarity diagnostic step independently of the Armijo trial. Otherwise it uses the initial step.

Set `warm_start_step` to `true` to start each subsequent Armijo search from the last accepted step. It defaults to `false`.

## Acquisition design

The `acquisition_design` section supplies `grid`, `pose`, non-negative FP32 `attenuation` in `mm^-1`, six positive `parameter_scales`, `samples_per_ray`, `max_candidate_pixels` and `current_information_description`. Scales are ordered translation x/y/z then rotation x/y/z, with units `(mm, mm, mm, rad, rad, rad)`. `prior_precision` names a symmetric positive-definite FP64 `(6,6)` array in `dimensionless` scaled coordinates. `targets_object_mm` names 1–256 FP64 object-frame points of shape `(N,3)` in `mm`.

Supply positive `maximum_cost`, a `cost_unit` of `mAs`, `s` or `expected_detector_photons`, and named `candidates`. Each candidate contains a unique non-empty `name`, `geometry`, `open_beam`, positive `cost` and `feasibility_description`. Beam images are non-negative FP32 in `photons/pixel`, with at least one illuminated pixel. At least one candidate must fit the budget. Photon-budget costs count expected detector photons and must match the open-beam sum within relative tolerance `1e-6`. Absorbed dose requires a separate physical calculation.

Optional `precision` defaults to `float32` and also accepts `float64`. `integration` defaults to `midpoint` and also accepts `cell_gauss`. `maximum_precision_condition` defaults to `1e12` and must be at least one. `rank_relative_tolerance` defaults to `1e-10` and must lie strictly between zero and one.

See [application drivers](drivers.md) for execution and outputs. Use the [operator contracts](../reference/operators.md) to check the units and storage of prepared arrays.
