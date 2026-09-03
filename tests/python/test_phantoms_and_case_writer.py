"""Independent chords, physical sampling and case-file round trips."""

import importlib
import math
from pathlib import Path
from typing import Any, cast

import pytest

from dpt.contracts import ContractError
from dpt.examples._common import ExampleInputError, load_case
from dpt.examples.make_case import CaseArray, write_case
from dpt.geometry import Vector3
from dpt.phantoms import Ellipsoid, circular_cone_beam, sample_ellipsoids
from dpt.volumes import GridSpec

np: Any = importlib.import_module("numpy")


@pytest.mark.parametrize("offset", [0.0, 0.3, 0.9, 1.0, 1.1])
@pytest.mark.parametrize("distance", [3.0, 1.0e9])
def test_sphere_chord_against_circle_identity(offset: float, distance: float) -> None:
    sphere = Ellipsoid((1.0, 1.0, 1.0), 0.3)
    expected = 0.6 * math.sqrt(max(0.0, 1.0 - offset**2))
    assert sphere.optical_depth((-distance, offset, 0), (distance, offset, 0)) == pytest.approx(
        expected, rel=2e-15, abs=1e-15
    )


def test_oriented_ellipsoid_and_clipped_segments() -> None:
    model = Ellipsoid((3.0, 2.0, 1.0), 0.2, (1.0, 2.0, 3.0), (0, -1, 0, 1, 0, 0, 0, 0, 1))
    assert model.optical_depth((1, -10, 3), (1, 10, 3)) == pytest.approx(1.2)
    assert model.optical_depth((1, 2, 3), (1, 10, 3)) == pytest.approx(0.6)
    assert model.optical_depth((1, 2, 3), (1, 3, 3)) == pytest.approx(0.2)
    assert model.optical_depth((1, 2, 3), (1, 2, 3)) == 0
    assert model.optical_depth((10, 2, 3), (20, 2, 3)) == 0


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_sampling_rotated_anisotropic_grid_and_additive_overlap(dtype: str) -> None:
    grid = GridSpec((3, 4, 5), (0.7, 1.3, 0.6), (-1.0, -2.0, -0.5), (0, -1, 0, 1, 0, 0, 0, 0, 1))
    models = (Ellipsoid((2, 1, 1), 0.1), Ellipsoid((1, 2, 2), 0.2, (0.2, -0.4, 0.1)))
    measured = sample_ellipsoids(grid, models, dtype=dtype)
    expected = np.zeros(grid.shape, dtype=dtype)
    for z in range(3):
        for y in range(4):
            for x in range(5):
                world = (-1.0 - 1.3 * y, -2.0 + 0.7 * x, -0.5 + 0.6 * z)
                expected[z, y, x] = sum(
                    model.attenuation_mm_inverse
                    for model in models
                    if sum(
                        ((world[i] - model.centre_mm[i]) / model.axes_mm[i]) ** 2 for i in range(3)
                    )
                    <= 1
                )
    np.testing.assert_array_equal(measured, expected)
    assert measured.flags.c_contiguous and measured.dtype == np.dtype(dtype)


def test_circular_panels_have_centred_pixels_and_exact_chief_chords() -> None:
    centre = (1.0, 2.0, 3.0)
    panels = circular_cone_beam(
        [0, math.pi / 2, math.pi],
        source_radius_mm=50,
        detector_radius_mm=30,
        shape_rows_columns=(3, 5),
        spacing_column_row_mm=(0.7, 1.2),
        centre_mm=centre,
    )
    sphere = Ellipsoid((4, 4, 4), 0.05, centre)
    for panel in panels:
        endpoint = tuple(
            panel.origin_mm[i] + 2 * 0.7 * panel.u[i] + 1.2 * panel.v[i] for i in range(3)
        )
        assert math.dist(panel.source_mm, centre) == pytest.approx(50)
        assert math.dist(endpoint, centre) == pytest.approx(30)
        assert sphere.optical_depth(panel.source_mm, cast(Vector3, endpoint)) == pytest.approx(0.4)


def test_case_writer_round_trip_and_input_identity(tmp_path: Path) -> None:
    values = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    path = write_case(
        tmp_path / "inputs",
        {"experiment": {"seed": 41}},
        {"volume": CaseArray(values, "mm^-1", "analytic test", "Apache-2.0")},
    )
    case = load_case(path)
    np.testing.assert_array_equal(case.array("volume", shape=(2, 3, 4), units="mm^-1"), values)
    assert case.config["experiment"] == {"seed": 41}
    with pytest.raises(ExampleInputError, match="already exists"):
        write_case(path.parent, {}, {"volume": CaseArray(values, "mm^-1", "test", "Apache-2.0")})
    values[0, 0, 0] = 99
    # The saved bytes are independent of later caller mutations.
    assert float(case.array("volume", units="mm^-1")[0, 0, 0]) == 0


@pytest.mark.parametrize(
    "values", [np.array([math.nan]), np.ones(3, dtype=complex), np.arange(12).reshape(3, 4).T]
)
def test_bad_case_is_rejected_before_directory_creation(values: Any, tmp_path: Path) -> None:
    destination = tmp_path / "bad"
    with pytest.raises(ExampleInputError):
        write_case(destination, {}, {"input": CaseArray(values, "mm", "test", "Apache-2.0")})
    assert not destination.exists()


def test_phantom_rejects_invalid_axes() -> None:
    with pytest.raises(ContractError):
        Ellipsoid((0, 1, 1), 0.2)
