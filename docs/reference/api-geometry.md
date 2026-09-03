# Geometry and physical inputs

Host records define physical coordinates, acquisitions and supplied material data. Device geometry expands compact panel descriptions on the owning stream. Imaging readers retain voxel values, coordinates and ingestion metadata. These modules import without CUDA.

## `dpt.geometry`


::: dpt.geometry.RigidTransform

::: dpt.geometry.DetectorGeometry

::: dpt.geometry.RayBundleGeometry

::: dpt.geometry.ParallelGeometry

::: dpt.geometry.CurvedDetectorGeometry

::: dpt.geometry.ConeBeamIntrinsics

::: dpt.geometry.BatchedGeometry

::: dpt.geometry.AstraVectorGeometry

::: dpt.geometry.compose_pose

::: dpt.geometry.right_jacobian_se3

::: dpt.geometry.chart_gradient

## `dpt.geometry_device`

See [multiview acquisitions](../guides/batched-geometry.md) for panel layout, object-frame mappings and ASTRA vector exchange.

::: dpt.geometry_device.DeviceRayBundleGeometry
    options:
      show_source: true

::: dpt.geometry_device.prepare_geometry
    options:
      show_source: true

## `dpt.projective`

`ProjectionMatrix` stores three homogeneous rows mapping world mm to column, row and depth. Pixel coordinates refer to pixel centres. `from_detector` exports a flat metric panel, and `project` returns continuous `(row, column)` coordinates at positive depth.

`to_detector(shape=..., detector_distance_mm=..., positive_depth_point_mm=...)` recovers the source, first-pixel centre, orthogonal detector axes and pixel spacings. Supply the perpendicular source-to-panel distance and a point in the physical source-to-panel hemisphere. The [matrix round-trip example](../guides/batched-geometry.md#recover-a-metric-detector-from-a-projection-matrix) makes these two metric/sign choices explicit.


::: dpt.projective.ProjectionMatrix
    options:
      show_source: true

## `dpt.tigre_geometry`

`TigreConeGeometry` owns a cone acquisition independently of the native TIGRE runtime. Its named fields distinguish `(rows, columns)` panel layouts, `(Z, Y, X)` origin offsets, `(V, U)` detector offsets, XYZ detector rotation and ZYZ acquisition angles. Distances use mm and angles use radians.

`to_dpt(world_from_tigre=...)` maps native physical rays into an explicit DPT world frame. `from_dpt` accepts the acquisition angles, origin offsets and centre-of-rotation corrections defining the source orbit. `from_tigre` copies existing native geometry values. `to_tigre` loads NumPy/TIGRE and creates a new geometry plus owned acquisition-angle array. The [TIGRE exchange example](../guides/batched-geometry.md#exchange-tigre-cone-geometry) includes tilted panels and a complete named-frame round trip.


::: dpt.tigre_geometry.TigreConeGeometry
    options:
      show_source: true

## `dpt.imaging_io`

See [imaging volumes](../guides/imaging-volumes.md) for sampled spheres, NIfTI,
classic DICOM stacks and enhanced frames. `ImagingMetadata` retains the
coordinate frame, units, source hashes and selected value transforms.
`read_dicom_series` selects modality or real-world mapping explicitly when
both are supplied.

::: dpt.imaging_io.ImagingMetadata

::: dpt.imaging_io.ImagingVolume

::: dpt.imaging_io.sphere_phantom
    options:
      show_source: true

::: dpt.imaging_io.read_nifti
    options:
      show_source: true

::: dpt.imaging_io.read_dicom_series
    options:
      show_source: true

## `dpt.volumes`


::: dpt.volumes.GridSpec

## `dpt.phantoms`

Use [synthetic phantoms](../guides/phantoms.md) for exact finite-segment ellipsoid
integrals, sampled z/y/x volumes and circular cone views. The
[numerical quickstart](../getting-started/quickstart.md) fits generated sphere
observations without external files.


::: dpt.phantoms.Ellipsoid

::: dpt.phantoms.sample_ellipsoids

::: dpt.phantoms.circular_cone_beam

## `dpt.materials`


::: dpt.materials.Provenance

::: dpt.materials.MaterialTable

::: dpt.materials.MaterialBasis

::: dpt.materials.validate_energy_grid

## `dpt.contracts`


::: dpt.contracts.ContractError

::: dpt.contracts.TrialDomainError

::: dpt.contracts.NumericalError
