"""Explicit finite-aperture primary counts and their matched first-order VJP.

Every pixel is a weighted bundle of finite rays. Counts average transmission,
with the supplied beam denoting the whole pixel's incident expectation. No
focal-spot law, spectral distribution, dark count or spatial PSF is inferred.
"""

from __future__ import annotations

# pyright: reportPrivateUsage=false
from dataclasses import dataclass, field
from itertools import pairwise
from math import fsum
from typing import TYPE_CHECKING, Any, Literal, cast

from dpt._runtime import DeviceContext, load_kernels, require_no_tape
from dpt.contracts import ContractError, finite_tuple, integer
from dpt.geometry import DetectorGeometry, RayBundleGeometry, Vector3
from dpt.projection import (
    ProjectionSpec,
    ProjectionWorkspace,
    prepare_projection,
    project_optical_depth,
    projection_vjp,
)
from dpt.volumes import GridSpec

if TYPE_CHECKING:
    import warp as wp


@dataclass(frozen=True, slots=True)
class ApertureGeometry:
    """Explicit ray groups and normalized nonnegative incident/response weights.

    Each non-empty group produces one pixel; weights are supplied physical
    assumptions and are never silently renormalized. Rays share one object pose.
    """

    rays: RayBundleGeometry
    pixel_offsets: tuple[int, ...]
    weights: tuple[float, ...]

    def __post_init__(self) -> None:
        if not isinstance(cast(object, self.rays), RayBundleGeometry):
            raise ContractError("aperture rays must be RayBundleGeometry")
        offsets = self.pixel_offsets
        if (
            not isinstance(cast(object, offsets), tuple)
            or len(offsets) < 2
            or any(type(offset) is not int for offset in offsets)
            or offsets[0] != 0
            or offsets[-1] != self.rays.rays
            or any(left >= right for left, right in pairwise(offsets))
        ):
            raise ContractError("pixel_offsets must partition all rays into non-empty pixels")
        weights = finite_tuple(self.weights, "aperture weights", length=self.rays.rays)
        for start, end in pairwise(offsets):
            if abs(fsum(weights[start:end]) - 1.0) > 1e-12:
                raise ContractError("aperture weights must sum to one within 1e-12 in every pixel")
        object.__setattr__(self, "weights", weights)

    @property
    def pixels(self) -> int:
        return len(self.pixel_offsets) - 1

    @classmethod
    def stratified(
        cls,
        detector: DetectorGeometry,
        *,
        samples_rows: int,
        samples_columns: int,
    ) -> ApertureGeometry:
        """Uniform area/incident weighting at deterministic rectangular stratum centres.

        The uniform response and illumination assumption is explicit. Increasing
        strata refines the aperture integral; it does not alter field quadrature.
        """
        if not isinstance(cast(object, detector), DetectorGeometry):
            raise ContractError("stratified aperture requires DetectorGeometry")
        integer(samples_rows, "samples_rows", minimum=1)
        integer(samples_columns, "samples_columns", minimum=1)
        count = samples_rows * samples_columns
        if detector.pixels * count > 2**31 - 1:
            raise ContractError("aperture exceeds signed 32-bit ray indexing")
        sources: list[Vector3] = []
        endpoints: list[Vector3] = []
        offsets = [0]
        for row in range(detector.shape[0]):
            for column in range(detector.shape[1]):
                centre = detector.pixel_centre(row, column)
                for subrow in range(samples_rows):
                    for subcolumn in range(samples_columns):
                        du = ((subcolumn + 0.5) / samples_columns - 0.5) * detector.spacing_mm[0]
                        dv = ((subrow + 0.5) / samples_rows - 0.5) * detector.spacing_mm[1]
                        sources.append(detector.source_mm)
                        endpoints.append(
                            cast(
                                Vector3,
                                tuple(
                                    centre[i] + du * detector.u[i] + dv * detector.v[i]
                                    for i in range(3)
                                ),
                            )
                        )
                offsets.append(len(sources))
        return cls(
            RayBundleGeometry(tuple(sources), tuple(endpoints)),
            tuple(offsets),
            (1.0 / count,) * len(sources),
        )


@dataclass(slots=True)
class ApertureWorkspace:
    """One prepared stream; subray depth and cotangent buffers belong to the caller."""

    geometry: ApertureGeometry
    projection: ProjectionWorkspace
    depths: Any
    depth_cotangents: Any
    _offsets: Any = field(repr=False)
    _weights: Any = field(repr=False)
    _kernels: Any = field(repr=False)

    @property
    def context(self) -> DeviceContext:
        return self.projection.context

    @property
    def scratch_bytes(self) -> int:
        return (
            self.projection.scratch_bytes
            + 4 * (self.geometry.pixels + 1)
            + 8 * self.geometry.rays.rays
        )

    @property
    def host_bytes(self) -> int:
        return self.projection.host_bytes

    def check_status(self) -> None:
        self.projection.check_status()

    def clear_status(self) -> None:
        self.projection.clear_status()

    def consume_status(self) -> None:
        self.projection.consume_status()


def prepare_aperture_projection(
    grid: GridSpec,
    geometry: ApertureGeometry,
    spec: ProjectionSpec,
    *,
    depths: wp.array[wp.float64],
    depth_cotangents: wp.array[wp.float64] | None,
    device: str | wp.Device = "cuda:0",
    stream: wp.Stream | None = None,
) -> ApertureWorkspace:
    """Prepare metadata; callers provide flat FP64 subray depths and cotangents.
    ``depth_cotangents`` binds FP64 (subrays,) derivative scratch.
    The caller owns
    both buffers and retains them for the workspace lifetime.
    """
    # Runtime contracts validate the dynamically loaded Warp boundary.
    _api_depths: Any = cast(Any, depths)
    _api_depth_cotangents: Any = cast(Any, depth_cotangents)
    if not isinstance(cast(object, geometry), ApertureGeometry):
        raise ContractError("aperture geometry must be ApertureGeometry")
    if spec.precision != "float64" or spec.field_domain != "physical":
        raise ContractError("aperture counts require physical fields and float64 depth precision")
    if spec.storage_precision != "float32":
        raise ContractError("aperture projection requires float32 field storage")
    if spec.sampling_precision == "fast_float32":
        raise ContractError("aperture projection supports float64 or mixed_float32 sampling")
    if _api_depth_cotangents is None:
        raise ContractError("depth_cotangents is required")
    projection = prepare_projection(grid, geometry.rays, spec, device=device, stream=stream)
    ctx, wp = projection.context, projection.context.wp
    ctx.array(_api_depths, "depths", dtype=wp.float64, shape=(geometry.rays.rays,))
    ctx.array(
        _api_depth_cotangents, "depth_cotangents", dtype=wp.float64, shape=(geometry.rays.rays,)
    )
    ctx.disjoint([], [("depths", _api_depths), ("depth_cotangents", _api_depth_cotangents)])
    with ctx.scope():
        offsets = wp.array(geometry.pixel_offsets, dtype=wp.int32, device=ctx.device)
        weights = wp.array(geometry.weights, dtype=wp.float64, device=ctx.device)
    return ApertureWorkspace(
        geometry,
        projection,
        _api_depths,
        _api_depth_cotangents,
        offsets,
        weights,
        load_kernels("dpt.kernels.aperture"),
    )


def _inputs(
    mu: Any,
    pose: Any,
    beam: Any,
    workspace: ApertureWorkspace,
    stream: wp.Stream | None,
    seed: Any = None,
) -> None:
    ctx, wp = workspace.context, workspace.context.wp
    ctx.assert_stream(stream)
    ctx.array(mu, "mu", dtype=wp.float32, shape=(workspace.projection.grid.voxels,))
    ctx.array(pose, "pose", dtype=wp.float64, shape=(12,))
    ctx.array(beam, "beam", dtype=wp.float64, shape=(workspace.geometry.pixels,))
    if seed is not None:
        ctx.array(seed, "adj_counts", dtype=wp.float64, shape=(workspace.geometry.pixels,))


def _validate(
    mu: Any, pose: Any, beam: Any, workspace: ApertureWorkspace, seed: Any = None
) -> None:
    workspace.projection.validate_inputs(mu, pose)
    wp = workspace.context.wp
    wp.launch(
        workspace._kernels._get_input_check(seed is not None),
        dim=workspace.geometry.pixels,
        inputs=[beam, seed if seed is not None else beam],
        outputs=[workspace.projection._status],
        stream=workspace.projection.stream,
        device=workspace.context.device,
        record_tape=False,
    )
    workspace.check_status()


def project_aperture_counts(
    mu: wp.array[wp.float32],
    pose: wp.array[wp.float64],
    beam: wp.array[wp.float64] | None,
    *,
    workspace: ApertureWorkspace,
    out_counts: wp.array[wp.float64],
    stream: wp.Stream | None = None,
    validate: bool = True,
) -> None:
    """Overwrite ``counts[p] = beam[p] sum_{r in pixel p} w[r] exp(-L[r])``.

    ``L[r]`` is the dimensionless optical depth along subray ``r``. ``mu`` is
    a finite nonnegative binary32 ``(nz*ny*nx,)`` attenuation field in mm^-1, with
    x varying fastest. ``pose`` is binary64 ``(12,)`` containing a row-major
    rotation followed by translation in mm. ``beam`` is a required binary64
    ``(P,)`` array of finite nonnegative whole-pixel incident expectations,
    and the caller-owned ``out_counts`` is binary64 ``(P,)`` in expected photons.
    The workspace's caller-owned depth/cotangent scratch has extent ``(R,)``
    for the total number of subrays. Geometry weights sum to one within each
    pixel, so transmission is averaged before multiplication by the beam.

    The inputs and output must occupy disjoint contiguous storage on the
    workspace device. Calls overwrite the output and depth scratch on the
    owning stream, leave inputs unchanged and return ``None``. Checked calls
    validate before writes and deliver the completion status. With
    ``validate=False``, call ``workspace.check_status()`` before consuming
    outputs. Dark counts and spatial response compose in count space, and this
    operator runs outside an ambient autodiff tape.

    Example::

        import math
        import warp as wp
        from dpt.aperture import ApertureGeometry, prepare_aperture_projection
        from dpt.geometry import RayBundleGeometry, RigidTransform
        from dpt.projection import ProjectionSpec
        from dpt.volumes import GridSpec

        grid = GridSpec((1, 1, 2), (1.0, 1.0, 1.0))
        rays = RayBundleGeometry(
            ((0.0, 0.0, -2.0), (1.0, 0.0, -2.0)),
            ((0.0, 0.0, 2.0), (1.0, 0.0, 2.0)),
        )
        geometry = ApertureGeometry(rays, (0, 2), (0.25, 0.75))
        workspace = prepare_aperture_projection(
            grid, geometry, ProjectionSpec(17, precision="float64", integration="cell_gauss"),
            depths=wp.empty(2, dtype=wp.float64, device="cpu"),
            depth_cotangents=wp.empty(2, dtype=wp.float64, device="cpu"), device="cpu",
        )
        mu = wp.array([1.0, 3.0], dtype=wp.float32, device="cpu")
        pose = wp.array(RigidTransform().packed(), dtype=wp.float64, device="cpu")
        beam = wp.array([100.0], dtype=wp.float64, device="cpu")

        from dpt.aperture import project_aperture_counts
        counts = wp.empty(1, dtype=wp.float64, device="cpu")
        project_aperture_counts(mu, pose, beam, workspace=workspace, out_counts=counts)
        expected = 100.0 * (0.25 * math.exp(-1.0) + 0.75 * math.exp(-3.0))
        assert math.isclose(float(counts.numpy()[0]), expected, rel_tol=1e-12)

    """
    # Runtime contracts validate the dynamically loaded Warp boundary.
    _api_mu: Any = cast(Any, mu)
    _api_pose: Any = cast(Any, pose)
    _api_beam: Any = cast(Any, beam)
    _api_out_counts: Any = cast(Any, out_counts)
    require_no_tape()
    _inputs(_api_mu, _api_pose, _api_beam, workspace, stream)
    ctx, wp = workspace.context, workspace.context.wp
    ctx.array(_api_out_counts, "out_counts", dtype=wp.float64, shape=(workspace.geometry.pixels,))
    ctx.disjoint(
        [("mu", _api_mu), ("pose", _api_pose), ("beam", _api_beam)],
        [
            ("depths", workspace.depths),
            ("depth_cotangents", workspace.depth_cotangents),
            ("out_counts", _api_out_counts),
        ],
    )
    if validate:
        _validate(_api_mu, _api_pose, _api_beam, workspace)
    project_optical_depth(
        _api_mu, _api_pose, workspace=workspace.projection, out_L=workspace.depths, validate=False
    )
    wp.launch(
        workspace._kernels.forward,
        dim=workspace.geometry.pixels,
        inputs=[workspace.depths, _api_beam, workspace._offsets, workspace._weights],
        outputs=[_api_out_counts, workspace.projection._status],
        stream=workspace.projection.stream,
        device=workspace.context.device,
        record_tape=False,
    )
    if validate:
        workspace.check_status()


def aperture_projection_vjp(
    mu: wp.array[wp.float32],
    pose: wp.array[wp.float64],
    beam: wp.array[wp.float64] | None,
    *,
    cotangent: wp.array[wp.float64] | None,
    workspace: ApertureWorkspace,
    out_pose_tangent: wp.array[wp.float64] | None = None,
    out_pose_matrix: wp.array[wp.float64] | None = None,
    out_mu: wp.array[wp.float32] | wp.array[wp.float64] | None = None,
    out_beam: wp.array[wp.float64] | None = None,
    volume_gradient_precision: Literal["float32", "float64"] = "float32",
    stream: wp.Stream | None = None,
    validate: bool = True,
    accumulate: bool = False,
) -> None:
    """Apply ``bar_L[r] = -g[p] beam[p] w[r] exp(-L[r])`` to the ray VJP.

    For ``g=cotangent``, the beam derivative is
    ``bar_beam[p] = g[p] sum_{r in pixel p} w[r] exp(-L[r])``. ``mu`` is flat
    finite nonnegative binary32 ``(nz*ny*nx,)`` attenuation in mm^-1 with x varying fastest,
    ``pose`` is binary64 ``(12,)`` row-major rotation then translation in mm,
    and required nonnegative ``beam`` and signed ``cotangent`` are finite
    binary64 ``(P,)`` arrays.
    The workspace recomputes its binary64 ``(R,)`` depth and depth-cotangent
    scratch from the current inputs.

    Request at least one caller-owned destination. ``out_mu`` has the field's
    extent and the selected ``volume_gradient_precision`` (binary32 by default
    or binary64), with units of the seeded scalar per mm^-1. ``out_beam`` is
    binary64 ``(P,)`` per incident photon. ``out_pose_tangent`` is binary64
    ``(6,)`` for ``pose @ exp(delta^)`` in the order
    ``(tx,ty,tz,rx,ry,rz)``, with derivatives per mm/rad.
    ``out_pose_matrix`` is binary64 ``(12,)`` for stored rotation/translation
    entries. Choose one pose representation and declare requested field/pose
    variables active at preparation.

    The default overwrites requested derivatives, while ``accumulate=True``
    adds to them. All inputs, scratch and destinations must be disjoint and
    belong to the workspace device. Products preserve attenuation range before
    rounding, including representable derivatives of underflowed counts.
    Calls return ``None`` on the owning stream, and unchecked calls require
    ``workspace.check_status()`` before output use. The supplied covector
    defines the seeded scalar, without an inferred objective or observation law.

    Example::

        import math
        import warp as wp
        from dpt.aperture import ApertureGeometry, prepare_aperture_projection
        from dpt.geometry import RayBundleGeometry, RigidTransform
        from dpt.projection import ProjectionSpec
        from dpt.volumes import GridSpec

        grid = GridSpec((1, 1, 2), (1.0, 1.0, 1.0))
        rays = RayBundleGeometry(
            ((0.0, 0.0, -2.0), (1.0, 0.0, -2.0)),
            ((0.0, 0.0, 2.0), (1.0, 0.0, 2.0)),
        )
        geometry = ApertureGeometry(rays, (0, 2), (0.25, 0.75))
        workspace = prepare_aperture_projection(
            grid, geometry, ProjectionSpec(17, precision="float64", integration="cell_gauss"),
            depths=wp.empty(2, dtype=wp.float64, device="cpu"),
            depth_cotangents=wp.empty(2, dtype=wp.float64, device="cpu"), device="cpu",
        )
        mu = wp.array([1.0, 3.0], dtype=wp.float32, device="cpu")
        pose = wp.array(RigidTransform().packed(), dtype=wp.float64, device="cpu")
        beam = wp.array([100.0], dtype=wp.float64, device="cpu")

        from dpt.aperture import aperture_projection_vjp
        cotangent = wp.array([-1.5], dtype=wp.float64, device="cpu")
        beam_gradient = wp.empty(1, dtype=wp.float64, device="cpu")
        aperture_projection_vjp(
            mu, pose, beam, cotangent=cotangent, workspace=workspace, out_beam=beam_gradient,
        )
        expected = -1.5 * (0.25 * math.exp(-1.0) + 0.75 * math.exp(-3.0))
        assert math.isclose(float(beam_gradient.numpy()[0]), expected, rel_tol=1e-12)

    """
    # Runtime contracts validate the dynamically loaded Warp boundary.
    _api_mu: Any = cast(Any, mu)
    _api_pose: Any = cast(Any, pose)
    _api_beam: Any = cast(Any, beam)
    _api_cotangent: Any = cast(Any, cotangent)
    _api_out_pose_tangent: Any = cast(Any, out_pose_tangent)
    _api_out_pose_matrix: Any = cast(Any, out_pose_matrix)
    _api_out_mu: Any = cast(Any, out_mu)
    _api_out_beam: Any = cast(Any, out_beam)
    require_no_tape()
    _inputs(_api_mu, _api_pose, _api_beam, workspace, stream, _api_cotangent)
    if volume_gradient_precision not in ("float32", "float64"):
        raise ContractError("volume_gradient_precision must be float32 or float64")
    if _api_out_pose_tangent is not None and _api_out_pose_matrix is not None:
        raise ContractError("request either a right-tangent or a storage pose gradient")
    if (
        _api_out_pose_tangent is None
        and _api_out_pose_matrix is None
        and _api_out_mu is None
        and _api_out_beam is None
    ):
        raise ContractError("request at least one aperture gradient")
    ctx, wp = workspace.context, workspace.context.wp
    writes = [("depths", workspace.depths), ("depth_cotangents", workspace.depth_cotangents)]
    target = _api_out_pose_tangent if _api_out_pose_tangent is not None else _api_out_pose_matrix
    if target is not None:
        if not workspace.projection.spec.active_pose:
            raise ContractError("pose is fixed in this workspace")
        ctx.array(
            target,
            "pose gradient",
            dtype=wp.float64,
            shape=(6 if _api_out_pose_tangent is not None else 12,),
        )
        writes.append(("pose gradient", target))
    if _api_out_mu is not None:
        if not workspace.projection.spec.active_volume:
            raise ContractError("volume is fixed in this workspace")
        ctx.array(
            _api_out_mu,
            "out_mu",
            dtype=getattr(wp, volume_gradient_precision),
            shape=(workspace.projection.grid.voxels,),
        )
        writes.append(("out_mu", _api_out_mu))
    if _api_out_beam is not None:
        ctx.array(_api_out_beam, "out_beam", dtype=wp.float64, shape=(workspace.geometry.pixels,))
        writes.append(("out_beam", _api_out_beam))
    ctx.disjoint(
        [
            ("mu", _api_mu),
            ("pose", _api_pose),
            ("beam", _api_beam),
            ("cotangent", _api_cotangent),
        ],
        writes,
    )
    if validate:
        _validate(_api_mu, _api_pose, _api_beam, workspace, _api_cotangent)
    project_optical_depth(
        _api_mu, _api_pose, workspace=workspace.projection, out_L=workspace.depths, validate=False
    )
    wp.launch(
        workspace._kernels.get_vjp(_api_out_beam is not None, accumulate),
        dim=workspace.geometry.pixels,
        inputs=[
            workspace.depths,
            _api_beam,
            _api_cotangent,
            workspace._offsets,
            workspace._weights,
        ],
        outputs=[
            workspace.depth_cotangents,
            _api_out_beam if _api_out_beam is not None else workspace.projection._empty_partials,
            workspace.projection._status,
        ],
        stream=workspace.projection.stream,
        device=workspace.context.device,
        record_tape=False,
    )
    if target is not None or _api_out_mu is not None:
        projection_vjp(
            _api_mu,
            _api_pose,
            cotangent=workspace.depth_cotangents,
            workspace=workspace.projection,
            out_pose_tangent=_api_out_pose_tangent,
            out_pose_matrix=_api_out_pose_matrix,
            out_mu=_api_out_mu,
            volume_gradient_precision=volume_gradient_precision,
            validate=False,
            accumulate=accumulate,
        )
    if validate:
        workspace.check_status()
