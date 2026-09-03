"""Composite projectors reject unimplemented storage and sampling policies."""

from typing import Any, Literal, cast

import pytest

from dpt.aperture import ApertureGeometry, prepare_aperture_projection
from dpt.contracts import ContractError
from dpt.geometry import DetectorGeometry
from dpt.material_projection import prepare_material_projection
from dpt.projection import ProjectionSpec
from dpt.volumes import GridSpec


@pytest.mark.parametrize("storage", ["float16", "bfloat16"])
def test_composite_projectors_reject_half_storage_before_device_allocation(
    storage: Literal["float16", "bfloat16"],
) -> None:
    grid = GridSpec((1, 1, 1), (1.0, 1.0, 1.0))
    detector = DetectorGeometry(
        (-2.0, 0.0, 0.0),
        (2.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (1.0, 1.0),
        (1, 1),
    )
    spec = ProjectionSpec(3, precision="float64", storage_precision=storage)
    with pytest.raises(ContractError, match="float32 field storage"):
        prepare_material_projection(
            grid,
            detector,
            spec,
            materials=1,
            fields=cast(Any, None),
            out_paths=cast(Any, None),
        )
    aperture = ApertureGeometry.stratified(detector, samples_rows=1, samples_columns=1)
    with pytest.raises(ContractError, match="float32 field storage"):
        prepare_aperture_projection(
            grid, aperture, spec, depths=cast(Any, None), depth_cotangents=None
        )


def test_composite_projectors_reject_fast_sampling_before_device_allocation() -> None:
    grid = GridSpec((1, 1, 1), (1.0, 1.0, 1.0))
    detector = DetectorGeometry(
        (-2.0, 0.0, 0.0),
        (2.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (1.0, 1.0),
        (1, 1),
    )
    spec = ProjectionSpec(
        3,
        precision="float64",
        active_pose=False,
        sampling_precision="fast_float32",
    )
    with pytest.raises(ContractError, match="sampling"):
        prepare_material_projection(
            grid,
            detector,
            spec,
            materials=1,
            fields=cast(Any, None),
            out_paths=cast(Any, None),
        )
    aperture = ApertureGeometry.stratified(detector, samples_rows=1, samples_columns=1)
    with pytest.raises(ContractError, match="sampling"):
        prepare_aperture_projection(
            grid, aperture, spec, depths=cast(Any, None), depth_cotangents=None
        )
