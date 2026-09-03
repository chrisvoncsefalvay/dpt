"""Device acceptance and slab VJPs against independent geometry/physics references."""

from __future__ import annotations

import importlib
import math
from dataclasses import replace
from typing import Any, Literal, cast

import numpy as np
import pytest

from dpt.contracts import ContractError, NumericalError
from dpt.detector_physics import (
    DetectorResponse,
    FirstInteractionModel,
    pixel_solid_angle,
    prepare_pixel_solid_angle,
    prepare_scintillator_response,
    scintillator_response,
    scintillator_response_vjp,
)
from dpt.geometry import (
    BatchedGeometry,
    CurvedDetectorGeometry,
    DetectorGeometry,
    ParallelGeometry,
    RigidTransform,
)
from dpt.materials import MaterialTable, Provenance
from helpers.detector_physics_oracles import (
    decimal_derivatives,
    decimal_rectangle_angle,
    decimal_response,
    surface_solid_angle,
)

wp: Any = importlib.import_module("warp")


def panel(*, curved: bool = False, offset: float = 1.3) -> Any:
    cls = CurvedDetectorGeometry if curved else DetectorGeometry
    return cls(
        (0.0, 0.0, 0.0),
        (0.0 if curved else offset, -1.8, 7.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (1.4, 0.6),
        (3, 5),
    )


@pytest.mark.parametrize("curved", [False, True])
def test_solid_angles_match_independent_surface_integrals(curved: bool, warp_device: str) -> None:
    original = panel(curved=curved)
    rotation = (0.0, 0.0, 1.0, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0)
    transform = RigidTransform(rotation, (100.0, -12.0, 3.0))
    moved = replace(
        original,
        source_mm=transform.point(original.source_mm),
        origin_mm=transform.point(original.origin_mm),
        u=transform.direction(original.u),
        v=transform.direction(original.v),
    )
    geometry = BatchedGeometry((original, moved))
    workspace = prepare_pixel_solid_angle(geometry, device=warp_device)
    output = wp.empty(geometry.shape, dtype=wp.float64, device=warp_device)
    cosine = wp.empty_like(output)
    pixel_solid_angle(out_solid_angle=output, out_incidence_cosine=cosine, workspace=workspace)
    measured = output.numpy()
    expected = np.array(
        [
            [surface_solid_angle(view, row, col) for row in range(3) for col in range(5)]
            for view in geometry.views
        ]
    ).reshape(geometry.shape)
    np.testing.assert_allclose(measured, expected, rtol=4e-14, atol=1e-16)
    np.testing.assert_allclose(measured[0], measured[1], rtol=2e-14, atol=0)
    for vi, view in enumerate(geometry.views):
        assert not isinstance(view, ParallelGeometry)
        for row in range(3):
            for col in range(5):
                endpoint = np.asarray(view.pixel_centre(row, col))
                ray = endpoint - np.asarray(view.source_mm)
                normal = np.cross(view.u, view.v)
                if curved:
                    axis_height = np.dot(ray, view.v)
                    normal = ray - axis_height * np.asarray(view.v)
                    normal /= np.linalg.norm(normal)
                expected_cosine = abs(float(np.dot(normal, ray))) / np.linalg.norm(ray)
                assert cosine.numpy()[vi, row, col] == pytest.approx(expected_cosine, rel=1e-14)
    assert workspace.scratch_bytes == 108
    assert workspace.host_bytes == 4


def test_finite_rectangle_near_source_and_far_off_axis(warp_device: str) -> None:
    near = DetectorGeometry(
        (0.0, 0.0, -0.5), (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (4.0, 2.0), (1, 1)
    )
    far = replace(near, source_mm=(0.0, 0.0, -1000.0), origin_mm=(1e6, 2e6, 0.0))
    geometry = BatchedGeometry((near, far))
    workspace = prepare_pixel_solid_angle(geometry, device=warp_device)
    angle = wp.empty(geometry.shape, dtype=wp.float64, device=warp_device)
    cosine = wp.empty_like(angle)
    pixel_solid_angle(out_solid_angle=angle, out_incidence_cosine=cosine, workspace=workspace)
    analytic = 4 * math.atan(2 * 1 / (0.5 * math.sqrt(0.5**2 + 2**2 + 1**2)))
    assert angle.numpy()[0, 0, 0] == pytest.approx(analytic, rel=1e-14)
    assert angle.numpy()[1, 0, 0] == pytest.approx(surface_solid_angle(far, 0, 0), rel=3e-14)


@pytest.mark.parametrize(
    "x,y,d,du,dv",
    [
        (0.0, 0.0, 1.0, 1e-10, 1e10),
        (0.0, 0.0, 1.0, 1e-8, 1e8),
        (0.0, 0.0, 1.0, 1e-150, 1e150),
        (10.0, 0.0, 1.0, 1e-16, 1e10),
        (-10.0, 0.0, 1.0, 1e-16, 1e10),
        (0.0, 10.0, 1.0, 1e10, 1e-16),
        (1e100, 1e100, 1.0, 1.0, 1.0),
        (0.0, 0.0, 2.0, 5e-324, 1e100),
        (5e-324, 0.0, 1.0, 1.5e-323, 1e100),
        (0.0, 5e-324, 1.0, 1e100, 1.5e-323),
    ],
)
def test_planar_exact_range_against_decimal_corner_integral(
    x: float, y: float, d: float, du: float, dv: float, warp_device: str
) -> None:
    geometry = DetectorGeometry(
        (0.0, 0.0, -d), (x, y, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (du, dv), (1, 1)
    )
    workspace = prepare_pixel_solid_angle(geometry, device=warp_device)
    angle = wp.empty((1, 1, 1), dtype=wp.float64, device=warp_device)
    cosine = wp.empty_like(angle)
    pixel_solid_angle(out_solid_angle=angle, out_incidence_cosine=cosine, workspace=workspace)
    expected = decimal_rectangle_angle(x, y, d, du, dv)
    assert expected > 0
    assert angle.numpy()[0, 0, 0] == pytest.approx(expected, rel=3e-14, abs=0)


@pytest.mark.parametrize("cosine,score", [(1e-200, 1.0), (1.0, 1e200)])
def test_device_slab_does_not_amplify_subnormal_depth_rounding(
    cosine: float, score: float, warp_device: str
) -> None:
    table = MaterialTable("analytic subnormal depth", (1e200,), (1e-200,), "mm^-1")
    model = (
        FirstInteractionModel("interaction-counting")
        if score == 1.0
        else FirstInteractionModel.ideal_full_energy()
    )
    workspace = prepare_scintillator_response(
        table,
        table.energies_kev,
        model=model,
        edge_side="above",
        max_pixels=1,
        precision="float64",
        device=warp_device,
    )
    t = wp.array([3e-124], dtype=wp.float64, device=warp_device)
    c = wp.array([cosine], dtype=wp.float64, device=warp_device)
    response = wp.empty((1, 1), dtype=wp.float64, device=warp_device)
    scintillator_response(t, c, out_response=response, workspace=workspace)
    expected = decimal_response(1e-200, 3e-124, cosine, score)
    assert response.numpy()[0, 0] == pytest.approx(expected, rel=3e-13, abs=0)


def test_device_slab_vjp_does_not_amplify_subnormal_numerator(warp_device: str) -> None:
    table = MaterialTable("analytic subnormal derivative", (1.0,), (1e-200,), "mm^-1")
    provenance = Provenance("analytic:score=3e-124", "d" * 64, "CC0-1.0", "Analytic fixture")
    model = FirstInteractionModel("supplied-deposition", (3e-124,), provenance)
    workspace = prepare_scintillator_response(
        table,
        table.energies_kev,
        model=model,
        edge_side="above",
        max_pixels=1,
        precision="float64",
        device=warp_device,
    )
    t = wp.zeros(1, dtype=wp.float64, device=warp_device)
    c = wp.array([1e-200], dtype=wp.float64, device=warp_device)
    seed = wp.ones((1, 1), dtype=wp.float64, device=warp_device)
    gt, gc = (wp.empty(1, dtype=wp.float64, device=warp_device) for _ in range(2))
    scintillator_response_vjp(
        t, c, cotangent=seed, out_thickness=gt, out_incidence_cosine=gc, workspace=workspace
    )
    expected = decimal_derivatives(1e-200, 0.0, 1e-200, 3e-124, 1.0)[0]
    assert gt.numpy()[0] == pytest.approx(expected, rel=3e-13, abs=0)
    assert gc.numpy()[0] == 0


def test_curved_angle_has_exact_arc_width_and_far_axial_tail(warp_device: str) -> None:
    base = panel(curved=True)
    far = replace(base, origin_mm=(0.0, 1e8, 7.0), shape=(1, 2), spacing_mm=(2.0, 1e-3))
    workspace = prepare_pixel_solid_angle(far, device=warp_device)
    angle = wp.empty((1, 1, 2), dtype=wp.float64, device=warp_device)
    cosine = wp.empty_like(angle)
    pixel_solid_angle(out_solid_angle=angle, out_incidence_cosine=cosine, workspace=workspace)
    assert angle.numpy()[0, 0, 0] == pytest.approx(surface_solid_angle(far, 0, 0), rel=5e-14)
    np.testing.assert_array_equal(angle.numpy()[0, 0, 0], angle.numpy()[0, 0, 1])
    assert angle.numpy().min() > 0


def test_solid_angle_shapes_alias_budget_and_point_source_contract(warp_device: str) -> None:
    geometry = panel()
    with pytest.raises(ContractError, match="max_owned_bytes"):
        prepare_pixel_solid_angle(geometry, device=warp_device, max_owned_bytes=59)
    with pytest.raises(ContractError, match="revolution"):
        prepare_pixel_solid_angle(
            replace(panel(curved=True), spacing_mm=(9.0, 0.6)), device=warp_device
        )
    parallel = ParallelGeometry(
        (0.0, 0.0, 1.0), (0.0, 0.0, 1.0), 1.0, (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (1.0, 1.0), (1, 1)
    )
    with pytest.raises(ContractError, match="parallel"):
        prepare_pixel_solid_angle(BatchedGeometry((parallel,)), device=warp_device)
    workspace = prepare_pixel_solid_angle(geometry, device=warp_device)
    angle = wp.empty((1, 3, 5), dtype=wp.float64, device=warp_device)
    with pytest.raises(ContractError, match="overlaps"):
        pixel_solid_angle(out_solid_angle=angle, out_incidence_cosine=angle, workspace=workspace)
    with pytest.raises(ContractError, match="shape"):
        pixel_solid_angle(
            out_solid_angle=angle,
            out_incidence_cosine=wp.empty((1, 5, 3), dtype=wp.float64, device=warp_device),
            workspace=workspace,
        )


def factors() -> tuple[MaterialTable, FirstInteractionModel]:
    table = MaterialTable(
        "analytic total scintillator fixture", (20.0, 40.0, 80.0), (0.5, 0.25, 0.125), "mm^-1"
    )
    model = FirstInteractionModel(
        "supplied-deposition",
        (5.0, 20.0, 24.0),
        Provenance(
            "analytic:conditional deposited energies",
            "c" * 64,
            "CC0-1.0",
            "Analytic fixture, no material dataset",
        ),
    )
    return table, model


@pytest.mark.parametrize("precision", ["float32", "float64"])
@pytest.mark.parametrize(
    "kind", ["interaction-counting", "supplied-deposition", "ideal-full-energy"]
)
def test_device_response_and_active_slab_vjp(
    precision: Literal["float32", "float64"],
    kind: Literal["interaction-counting", "supplied-deposition", "ideal-full-energy"],
    warp_device: str,
) -> None:
    table, supplied = factors()
    model = supplied if kind == "supplied-deposition" else FirstInteractionModel(kind)
    workspace = prepare_scintillator_response(
        table,
        table.energies_kev,
        model=model,
        edge_side="above",
        max_pixels=3,
        precision=precision,
        device=warp_device,
    )
    thickness = wp.array([0.0, 0.5, 2.0], dtype=wp.float64, device=warp_device)
    cosine = wp.array([1.0, 0.7, 0.2], dtype=wp.float64, device=warp_device)
    output = wp.empty((3, 3), dtype=workspace.dtype, device=warp_device)
    seed = wp.array(
        [[2.0, -1.0, 0.5], [-1.0, 3.0, -2.0], [4.0, 0.5, 2.0]],
        dtype=workspace.dtype,
        device=warp_device,
    )
    gt = wp.empty(3, dtype=wp.float64, device=warp_device)
    gc = wp.empty_like(gt)
    original = [x.numpy().copy() for x in (thickness, cosine, seed)]
    scintillator_response(thickness, cosine, out_response=output, workspace=workspace)
    expected = np.array(
        [
            [
                decimal_response(mu, t, c, score)
                for t, c in zip(original[0], original[1], strict=True)
            ]
            for mu, score in zip(table.coefficients, model.scores(table.energies_kev), strict=True)
        ]
    )
    np.testing.assert_allclose(
        output.numpy(), expected, rtol=8e-8 if precision == "float32" else 3e-15, atol=0
    )
    scintillator_response_vjp(
        thickness,
        cosine,
        cotangent=seed,
        out_thickness=gt,
        out_incidence_cosine=gc,
        workspace=workspace,
    )
    expected_g = np.zeros((2, 3))
    for p in range(3):
        for k, (mu, score) in enumerate(
            zip(table.coefficients, model.scores(table.energies_kev), strict=True)
        ):
            expected_g[:, p] += decimal_derivatives(
                mu, float(original[0][p]), float(original[1][p]), score, float(original[2][k, p])
            )
    np.testing.assert_allclose(gt.numpy(), expected_g[0], rtol=1e-13, atol=1e-15)
    np.testing.assert_allclose(gc.numpy(), expected_g[1], rtol=1e-13, atol=1e-15)
    for array, reference in zip((thickness, cosine, seed), original, strict=True):
        np.testing.assert_array_equal(array.numpy(), reference)
    # Fresh factor values, no stale forward image is needed by the reverse.
    thickness.assign(np.array([0.1, 0.5, 2.0]))
    scintillator_response_vjp(
        thickness,
        cosine,
        cotangent=seed,
        out_thickness=gt,
        out_incidence_cosine=gc,
        workspace=workspace,
    )
    assert gt.numpy()[0] != expected_g[0, 0]
    assert workspace.scratch_bytes == 52 and workspace.host_bytes == 4


def test_slab_gradients_match_finite_differences_of_defining_physics(warp_device: str) -> None:
    table, model = factors()
    workspace = prepare_scintillator_response(
        table,
        table.energies_kev,
        model=model,
        edge_side="above",
        max_pixels=2,
        precision="float64",
        device=warp_device,
    )
    thost, chost = np.array([0.4, 1.2]), np.array([0.6, 0.9])
    thickness = wp.array(thost, dtype=wp.float64, device=warp_device)
    cosine = wp.array(chost, dtype=wp.float64, device=warp_device)
    seedhost = np.array([[1.2, -0.7], [-0.2, 2.0], [0.5, -1.1]])
    seed = wp.array(seedhost, dtype=wp.float64, device=warp_device)
    gt, gc = (wp.empty(2, dtype=wp.float64, device=warp_device) for _ in range(2))
    scintillator_response_vjp(
        thickness,
        cosine,
        cotangent=seed,
        out_thickness=gt,
        out_incidence_cosine=gc,
        workspace=workspace,
    )

    def objective(t: np.ndarray, c: np.ndarray) -> float:
        result = np.array(
            [
                [score * (1 - np.exp(-mu * t[p] / c[p])) for p in range(2)]
                for mu, score in zip(
                    table.coefficients, model.scores(table.energies_kev), strict=True
                )
            ]
        )
        return float(np.sum(seedhost * result))

    step = 1e-5
    for name, values, grad in (("t", thost, gt.numpy()), ("c", chost, gc.numpy())):
        for p in range(2):
            plus, minus = values.copy(), values.copy()
            plus[p] += step
            minus[p] -= step
            observed = (
                (objective(plus, chost) - objective(minus, chost)) / (2 * step)
                if name == "t"
                else (objective(thost, plus) - objective(thost, minus)) / (2 * step)
            )
            assert observed == pytest.approx(grad[p], rel=1e-8, abs=1e-9)


@pytest.mark.parametrize(
    "bad", ["negative-thickness", "nan-thickness", "zero-cosine", "large-cosine", "nan-seed"]
)
def test_slab_invalid_values_sticky_status_and_explicit_recovery(
    bad: str, warp_device: str
) -> None:
    table, model = factors()
    workspace = prepare_scintillator_response(
        table, table.energies_kev, model=model, edge_side="above", max_pixels=1, device=warp_device
    )
    thickness = wp.array(
        [float("nan") if bad == "nan-thickness" else -1.0 if bad == "negative-thickness" else 0.5],
        dtype=wp.float64,
        device=warp_device,
    )
    cosine = wp.array(
        [0.0 if bad == "zero-cosine" else 1.001 if bad == "large-cosine" else 1.0],
        dtype=wp.float64,
        device=warp_device,
    )
    seed = wp.array(
        [[float("nan") if bad == "nan-seed" else 1.0], [1.0], [1.0]],
        dtype=wp.float32,
        device=warp_device,
    )
    gt, gc = (wp.empty(1, dtype=wp.float64, device=warp_device) for _ in range(2))
    scintillator_response_vjp(
        thickness,
        cosine,
        cotangent=seed,
        out_thickness=gt,
        out_incidence_cosine=gc,
        workspace=workspace,
        validate=False,
    )
    with pytest.raises(ContractError, match="invalid fixed"):
        workspace.check_status()
    thickness.fill_(0.5)
    cosine.fill_(1.0)
    seed.fill_(1.0)
    with pytest.raises(ContractError):
        scintillator_response_vjp(
            thickness,
            cosine,
            cotangent=seed,
            out_thickness=gt,
            out_incidence_cosine=gc,
            workspace=workspace,
        )
    workspace.clear_status()
    scintillator_response_vjp(
        thickness,
        cosine,
        cotangent=seed,
        out_thickness=gt,
        out_incidence_cosine=gc,
        workspace=workspace,
    )
    workspace.check_status()


def test_response_range_boundaries_and_deep_gradient_rescue(warp_device: str) -> None:
    table = MaterialTable("analytic weak response", (1e200,), (1e-200,), "mm^-1")
    workspace = prepare_scintillator_response(
        table,
        table.energies_kev,
        model=FirstInteractionModel("ideal-full-energy"),
        edge_side="above",
        max_pixels=1,
        precision="float64",
        device=warp_device,
    )
    thickness = wp.array([1e-200], dtype=wp.float64, device=warp_device)
    cosine = wp.array([1.0], dtype=wp.float64, device=warp_device)
    output = wp.empty((1, 1), dtype=wp.float64, device=warp_device)
    scintillator_response(thickness, cosine, out_response=output, workspace=workspace)
    assert output.numpy()[0, 0] == pytest.approx(1e-200, rel=2e-13)
    high = MaterialTable("analytic deep derivative", (1e200,), (1e200,), "mm^-1")
    deep = prepare_scintillator_response(
        high,
        high.energies_kev,
        model=FirstInteractionModel("ideal-full-energy"),
        edge_side="above",
        max_pixels=1,
        precision="float64",
        device=warp_device,
    )
    thickness.fill_(1500e-200)
    seed = wp.array([[1e200]], dtype=wp.float64, device=warp_device)
    gt, gc = (wp.empty(1, dtype=wp.float64, device=warp_device) for _ in range(2))
    scintillator_response_vjp(
        thickness, cosine, cotangent=seed, out_thickness=gt, out_incidence_cosine=gc, workspace=deep
    )
    expected = decimal_derivatives(1e200, 1500e-200, 1.0, 1e200, 1e200)
    assert gt.numpy()[0] == pytest.approx(expected[0], rel=5e-11)
    assert gc.numpy()[0] == pytest.approx(expected[1], rel=5e-11)
    narrow = prepare_scintillator_response(
        MaterialTable("analytic", (20.0,), (1.0,), "mm^-1"),
        (20.0,),
        model=FirstInteractionModel("interaction-counting"),
        edge_side="above",
        max_pixels=1,
        device=warp_device,
    )
    thickness.fill_(1e-50)
    with pytest.raises(NumericalError, match="represented"):
        scintillator_response(
            thickness,
            cosine,
            out_response=wp.empty((1, 1), dtype=wp.float32, device=warp_device),
            workspace=narrow,
        )


def test_slab_shape_alias_precision_and_budget_contracts(warp_device: str) -> None:
    table, model = factors()
    with pytest.raises(ContractError, match="max_owned_bytes"):
        prepare_scintillator_response(
            table,
            table.energies_kev,
            model=model,
            edge_side="above",
            max_pixels=2,
            device=warp_device,
            max_owned_bytes=55,
        )
    workspace = prepare_scintillator_response(
        table,
        table.energies_kev,
        model=model,
        edge_side="above",
        max_pixels=2,
        precision="float64",
        device=warp_device,
    )
    thickness, cosine = (wp.ones(2, dtype=wp.float64, device=warp_device) for _ in range(2))
    seed = wp.ones((3, 2), dtype=wp.float64, device=warp_device)
    gradient = wp.empty(2, dtype=wp.float64, device=warp_device)
    with pytest.raises(ContractError, match="overlaps"):
        scintillator_response_vjp(
            thickness,
            cosine,
            cotangent=seed,
            out_thickness=thickness,
            out_incidence_cosine=gradient,
            workspace=workspace,
        )
    with pytest.raises(ContractError, match="shape"):
        scintillator_response(
            thickness,
            cosine,
            out_response=wp.empty((2, 3), dtype=wp.float64, device=warp_device),
            workspace=workspace,
        )
    with pytest.raises(ContractError, match="dtype"):
        scintillator_response(
            thickness,
            cosine,
            out_response=wp.empty((3, 2), dtype=wp.float32, device=warp_device),
            workspace=workspace,
        )
    with pytest.raises(ContractError, match="boolean"):
        scintillator_response(
            thickness, cosine, out_response=seed, workspace=workspace, validate=cast(Any, 1)
        )


def test_acceptance_cosines_compose_with_slab_and_shared_recipe(warp_device: str) -> None:
    geometry = panel()
    acceptance = prepare_pixel_solid_angle(geometry, device=warp_device)
    angle, cosine = (wp.empty((1, 3, 5), dtype=wp.float64, device=warp_device) for _ in range(2))
    pixel_solid_angle(out_solid_angle=angle, out_incidence_cosine=cosine, workspace=acceptance)
    table, model = factors()
    workspace = prepare_scintillator_response(
        table,
        table.energies_kev,
        model=model,
        edge_side="above",
        max_pixels=15,
        precision="float64",
        device=warp_device,
    )
    thickness = wp.full(15, 0.7, dtype=wp.float64, device=warp_device)
    output = wp.empty((3, 15), dtype=wp.float64, device=warp_device)
    scintillator_response(thickness, cosine.flatten(), out_response=output, workspace=workspace)
    for p, c in enumerate(cosine.numpy().reshape(-1)):
        recipe = DetectorResponse(table, table.energies_kev, 0.7, model, "above", float(c))
        np.testing.assert_allclose(output.numpy()[:, p], recipe.values, rtol=1e-14)


def test_zero_attenuation_zero_score_and_saturation_have_defined_derivatives(
    warp_device: str,
) -> None:
    table = MaterialTable(
        "analytic transparent/saturating fixture",
        (20.0, 40.0, 80.0),
        (0.0, 0.5, 1e300),
        "mm^-1",
        interpolation="linear",
    )
    provenance = Provenance(
        "analytic:conditional scores=(1,0,3)keV", "d" * 64, "CC0-1.0", "Analytic fixture"
    )
    model = FirstInteractionModel("supplied-deposition", (1.0, 0.0, 3.0), provenance)
    workspace = prepare_scintillator_response(
        table,
        table.energies_kev,
        model=model,
        edge_side="above",
        max_pixels=1,
        precision="float64",
        device=warp_device,
    )
    thickness = wp.array([1e300], dtype=wp.float64, device=warp_device)
    cosine = wp.array([1.0], dtype=wp.float64, device=warp_device)
    output = wp.empty((3, 1), dtype=wp.float64, device=warp_device)
    seed = wp.ones((3, 1), dtype=wp.float64, device=warp_device)
    gt, gc = (wp.empty(1, dtype=wp.float64, device=warp_device) for _ in range(2))
    scintillator_response(thickness, cosine, out_response=output, workspace=workspace)
    scintillator_response_vjp(
        thickness,
        cosine,
        cotangent=seed,
        out_thickness=gt,
        out_incidence_cosine=gc,
        workspace=workspace,
    )
    np.testing.assert_array_equal(output.numpy(), [[0.0], [0.0], [3.0]])
    np.testing.assert_array_equal(gt.numpy(), [0.0])
    np.testing.assert_array_equal(gc.numpy(), [0.0])
