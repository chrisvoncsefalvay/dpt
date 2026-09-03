"""Owned detector response banks sharing a material-path projection."""

from __future__ import annotations

# Composition reuses the native workspace's complete factor-value checks.
# pyright: reportPrivateUsage=false
from contextlib import contextmanager
from dataclasses import dataclass, replace
from threading import Lock
from typing import TYPE_CHECKING, Any, Literal, cast

from dpt._runtime import (
    DeviceContext,
    StatusCheckpoint,
    StatusReadback,
    load_kernels,
    require_no_tape,
)
from dpt.contracts import ContractError, integer, raise_device_status
from dpt.geometry import DetectorGeometry, RayBundleGeometry
from dpt.material_projection import (
    material_projection_vjp,
    prepare_material_projection,
    project_material_paths,
)
from dpt.projection import ProjectionSpec
from dpt.spectral import SpectralSpec, prepare_spectral, spectral_signal, spectral_vjp
from dpt.volumes import GridSpec

if TYPE_CHECKING:
    from collections.abc import Generator

    import warp as wp

    FloatArray = wp.array[wp.float32] | wp.array[wp.float64]


def _flat_factor(
    values: wp.array[wp.float32],
    name: str,
    shape: tuple[int, ...],
    context: DeviceContext,
) -> wp.array[wp.float32]:
    if getattr(values, "ndim", 1) == len(shape):
        context.array(values, name, dtype=context.wp.float32, shape=shape, ndim=len(shape))
    else:
        length = 1
        for count in shape:
            length *= count
        context.array(values, name, dtype=context.wp.float32, shape=(length,))
    return cast("wp.array[wp.float32]", values.flatten())


@dataclass(frozen=True, slots=True, init=False)
class ResponseMatrixInputs:
    """Fixed factors for ``I[b,p] = sum_k w[k,p] R[b,k] exp(-mu[:,k]·A[:,p])``.

    Coefficients ``mu[M,K]`` use mm⁻¹. Weights contain integrated photon
    populations, shared ``[K]`` or pixel-specific ``[K,P]``. The response bank
    ``R[B,K]`` is shared across pixels in each output band. All factors have
    binary32 storage; the spectral specification selects signal precision,
    linear or natural-log output, units and active parameters.

    ``from_arrays`` validates independent copies once on the supplied device
    and stream. Keep the owned arrays unchanged and prepare another record
    when factors change. Original caller arrays can be modified afterwards.
    """

    spec: SpectralSpec
    bands: int
    pixels: int
    context: DeviceContext
    coefficients: wp.array[wp.float32]
    weights: wp.array[wp.float32]
    response: wp.array[wp.float32]
    _response_rows: tuple[wp.array[wp.float32], ...]

    def __init__(self) -> None:
        raise TypeError("prepare a response bank with ResponseMatrixInputs.from_arrays(...)")

    @classmethod
    def from_arrays(
        cls,
        spec: SpectralSpec,
        coefficients: wp.array[wp.float32],
        weights: wp.array[wp.float32],
        response: wp.array[wp.float32],
        *,
        pixels: int,
        device: str = "cuda:0",
        stream: Any = None,
        max_owned_bytes: int = 1 << 30,
    ) -> ResponseMatrixInputs:
        """Copy ``mu[M,K]``, ``w[K or K,P]`` and ``R[B,K]`` on one context.

        The response must retain its explicit band and energy axes. Coefficients
        and weights also accept contiguous flat storage in material/energy-major
        order. ``max_owned_bytes`` limits the fixed factor copies before any
        factor-sized allocation, including the temporary four-byte device
        diagnostic and four-byte host readback used during validation.
        """
        require_no_tape()
        if not spec.shared_response:
            raise ContractError("a response matrix requires shared_response=True")
        integer(pixels, "pixels", minimum=1)
        integer(max_owned_bytes, "max_owned_bytes", minimum=1, maximum=2**63 - 1)
        if getattr(response, "ndim", None) != 2:
            raise ContractError("response must have explicit shape (B,K)")
        bands = integer(int(cast(Any, response).shape[0]), "bands", minimum=1)
        integer(bands * spec.energies, "response bank length", minimum=1)
        integer(bands * pixels, "band image length", minimum=1)
        weight_shape = (spec.energies,) if spec.shared_weights else (spec.energies, pixels)
        owned_bytes = 4 * (
            spec.materials * spec.energies
            + spec.energies * (1 if spec.shared_weights else pixels)
            + bands * spec.energies
        )
        if owned_bytes + 8 > max_owned_bytes:
            raise ContractError("response matrix factors exceed max_owned_bytes")
        validation = prepare_spectral(
            replace(
                spec,
                active_paths=False,
                active_weights=False,
                active_response=False,
                active_coefficients=False,
                output_domain="linear",
            ),
            max_pixels=pixels,
            device=device,
            stream=stream,
        )
        context = validation.context
        sources = (
            _flat_factor(coefficients, "coefficients", (spec.materials, spec.energies), context),
            _flat_factor(weights, "weights", weight_shape, context),
            _flat_factor(response, "response", (bands, spec.energies), context),
        )
        with context.scope():
            owned = tuple(context.wp.clone(values) for values in sources)
            rows = tuple(
                cast("wp.array[wp.float32]", owned[2][b * spec.energies : (b + 1) * spec.energies])
                for b in range(bands)
            )
            validation.validate_inputs(
                owned[0], owned[1], rows[0], pixels=pixels, stream=context.stream
            )
            # Each band has the same spectral factor domain. Scan the entire
            # bank once; no factor validation repeats during evaluation.
            validation._check_values([(owned[2], True)])
        record = object.__new__(cls)
        values = (
            ("spec", spec),
            ("bands", bands),
            ("pixels", pixels),
            ("context", context),
            ("coefficients", owned[0].reshape((spec.materials, spec.energies))),
            ("weights", owned[1].reshape(weight_shape)),
            ("response", owned[2].reshape((bands, spec.energies))),
            ("_response_rows", rows),
        )
        for name, value in values:
            object.__setattr__(record, name, value)
        return record

    @property
    def storage_bytes(self) -> int:
        """Bytes in the owned binary32 coefficient, weight and response copies."""
        return 4 * (self.coefficients.size + self.weights.size + self.response.size)


def _projection_scratch_bytes(
    geometry: DetectorGeometry | RayBundleGeometry,
    projection: ProjectionSpec,
    context: DeviceContext,
) -> int:
    result = 8 + (48 * geometry.pixels if isinstance(geometry, RayBundleGeometry) else 0)
    if projection.active_pose:
        count = (geometry.pixels + 127) // 128 if context.device.is_cuda else geometry.pixels
        while True:
            result += max(1, count) * 12 * 8
            if count <= 1:
                break
            count = (count + 127) // 128
    return result


@dataclass(frozen=True, slots=True)
class ResponseMatrixGradients:
    """Cotangents of the most recent accepted response-bank forward.

    Fields retain binary32 ``[M,nz,ny,nx]`` storage. Pose has six binary64
    derivatives with respect to local-right increment coordinates: translation
    measured in mm followed by rotation in radians.
    Coefficients ``[M,K]``, shared weights ``[K]`` and responses ``[B,K]`` have
    binary64 cotangents; pixel-specific weights ``[K,P]`` retain binary32 storage.
    Fixed parameters have a value of ``None``. Buffers are pipeline-owned and
    overwritten by the next VJP.
    """

    fields: wp.array[wp.float32] | None
    pose: wp.array[wp.float64] | None
    coefficients: wp.array[wp.float64] | None
    weights: FloatArray | None
    response: wp.array[wp.float64] | None


@dataclass(frozen=True, slots=True)
class _CommonGradient:
    source: Any
    destination: Any
    limbs: Any
    source64: bool


class ResponseMatrixPipeline:
    """Evaluate a response bank from one set of material paths.

    ``forward`` computes material paths ``A[M,P]`` once, then evaluates each
    response row with the canonical linear or logarithmic spectral operator.
    The result always retains band and flattened pixel axes ``[B,P]``. Band
    arithmetic still requires ``B`` spectral evaluations; material ray traversal
    is independent of the number of detector bands.

    The pipeline owns a field and packed-pose snapshot on every forward. It
    executes serially on the inputs' device and stream, with persistent buffers.
    Results share pipeline storage and are overwritten by the next evaluation.

    Reverse evaluates each band's canonical VJP at its native representability
    boundary. It exactly sums every finite common cotangent, rounds once to its
    public storage type, then invokes one combined material-path reverse. A
    nonrepresentable native band derivative retains the native range failure,
    including when another band's derivative would cancel it. Response-row
    gradients remain independent. The reduction requires 40 bytes per common
    binary32 scalar or 272 bytes per common binary64 scalar, checked against
    ``max_owned_bytes`` before allocation. No pixel-by-energy tape is retained.
    """

    def __init__(
        self,
        grid: GridSpec,
        geometry: DetectorGeometry | RayBundleGeometry,
        projection: ProjectionSpec,
        inputs: ResponseMatrixInputs,
        *,
        field_domain: Literal["fractions", "nonnegative"] = "fractions",
        channel_execution: Literal["serial", "fused"] = "serial",
        max_owned_bytes: int = 1 << 30,
    ) -> None:
        require_no_tape()
        integer(max_owned_bytes, "max_owned_bytes", minimum=1, maximum=2**63 - 1)
        if inputs.pixels != geometry.pixels:
            raise ContractError("response matrix inputs and geometry require the same pixel count")
        if projection.precision != inputs.spec.precision:
            raise ContractError("projection and response matrix precision must match")
        if field_domain not in ("fractions", "nonnegative"):
            raise ContractError("field_domain must be fractions or nonnegative")
        if channel_execution not in ("serial", "fused"):
            raise ContractError("channel_execution must be serial or fused")
        if projection.field_domain != "physical" or projection.storage_precision != "float32":
            raise ContractError(
                "response matrix projection requires physical binary32 material fields"
            )
        if projection.sampling_precision == "fast_float32":
            raise ContractError(
                "response matrix projection supports float64 or mixed_float32 sampling"
            )
        if channel_execution == "fused" and projection.sampling_precision != "float64":
            raise ContractError("fused material projection requires strict sampling precision")
        if (projection.active_volume or projection.active_pose) and not inputs.spec.active_paths:
            raise ContractError("field or pose derivatives require active spectral paths")
        self.grid, self.geometry, self.projection, self.inputs = grid, geometry, projection, inputs
        self.context = inputs.context
        context = self.context
        context.assert_stream(context.stream)
        self._lock = Lock()
        self._has_forward = False
        materials, pixels, bands = inputs.spec.materials, inputs.pixels, inputs.bands
        integer(materials * grid.voxels, "material field length", minimum=1)
        integer(materials * pixels, "material path length", minimum=1)
        signal_bytes = 8 if inputs.spec.precision == "float64" else 4
        paths_count = materials * pixels if inputs.spec.active_paths else 0
        weights_count = (
            inputs.spec.energies * (1 if inputs.spec.shared_weights else pixels)
            if inputs.spec.active_weights
            else 0
        )
        coefficients_count = (
            materials * inputs.spec.energies if inputs.spec.active_coefficients else 0
        )
        weights_bytes = 8 if inputs.spec.shared_weights else 4
        self._aggregate_scratch_bytes = (
            paths_count * (signal_bytes + (272 if signal_bytes == 8 else 40))
            + weights_count * (weights_bytes + (272 if weights_bytes == 8 else 40))
            + coefficients_count * (8 + 272)
            + 4
        )
        for name, count, wide in (
            ("path", paths_count, signal_bytes == 8),
            ("weight", weights_count, weights_bytes == 8),
            ("coefficient", coefficients_count, True),
        ):
            integer(count * (34 if wide else 5), f"{name} accumulator length")
        parameters = (
            inputs.spec.energies * int(inputs.spec.active_weights and inputs.spec.shared_weights)
            + inputs.spec.energies * int(inputs.spec.active_response)
            + materials * inputs.spec.energies * int(inputs.spec.active_coefficients)
        )
        groups = min(inputs.spec.reduction_groups, max(1, (pixels + 255) // 256))
        self._base_storage_bytes = (
            inputs.storage_bytes
            + 4 * materials * grid.voxels
            + 96
            + signal_bytes * (2 * materials * pixels + 2 * bands * pixels)
            + (4 * materials * grid.voxels if projection.active_volume else 0)
            + (48 if projection.active_pose else 0)
            + weights_bytes * weights_count
            + 8 * coefficients_count
            + (8 * bands * inputs.spec.energies if inputs.spec.active_response else 0)
        )
        scratch_bytes = (
            _projection_scratch_bytes(geometry, projection, context)
            + 4
            + 8 * parameters * groups
            + (16 * pixels if inputs.spec.output_domain == "log" else 0)
            + self._aggregate_scratch_bytes
        )
        # Four status owners retain four bytes of host staging each. Their
        # temporary preparation allocations remain in the final workspaces.
        if self._base_storage_bytes + scratch_bytes + 16 > max_owned_bytes:
            raise ContractError("response matrix pipeline exceeds max_owned_bytes")
        self.max_owned_bytes = max_owned_bytes
        with context.scope():
            self._fields = cast(
                "wp.array[wp.float32]",
                context.wp.empty(
                    (materials, *grid.shape), dtype=context.wp.float32, device=context.device
                ),
            )
            self._pose = cast(
                "wp.array[wp.float64]",
                context.wp.empty(12, dtype=context.wp.float64, device=context.device),
            )
            self._paths = context.wp.empty(
                (materials, pixels),
                dtype=getattr(context.wp, projection.precision),
                device=context.device,
            )
            adj_paths = context.wp.empty_like(self._paths)
            self._images = cast(
                "FloatArray",
                context.wp.empty(
                    (bands, pixels),
                    dtype=getattr(context.wp, projection.precision),
                    device=context.device,
                ),
            )
            self._cotangents = context.wp.empty_like(self._images)
            images_flat = cast(Any, self._images).flatten()
            self._image_rows = tuple(
                images_flat[b * pixels : (b + 1) * pixels] for b in range(bands)
            )
            self._fields_gradient = (
                context.wp.empty_like(self._fields) if projection.active_volume else None
            )
            self._pose_gradient = (
                context.wp.empty(6, dtype=context.wp.float64, device=context.device)
                if projection.active_pose
                else None
            )
            self._status = context.wp.zeros(1, dtype=context.wp.int32, device=context.device)
            self._paths_common = self._common_gradient(adj_paths) if paths_count else None
            self._weights_gradient = (
                context.wp.empty(
                    (inputs.spec.energies,)
                    if inputs.spec.shared_weights
                    else (inputs.spec.energies, pixels),
                    dtype=context.wp.float64 if inputs.spec.shared_weights else context.wp.float32,
                    device=context.device,
                )
                if weights_count
                else None
            )
            self._weights_common = (
                self._common_gradient(self._weights_gradient) if weights_count else None
            )
            self._coefficients_gradient = (
                context.wp.empty(
                    (materials, inputs.spec.energies),
                    dtype=context.wp.float64,
                    device=context.device,
                )
                if coefficients_count
                else None
            )
            self._coefficients_common = (
                self._common_gradient(self._coefficients_gradient) if coefficients_count else None
            )
            self._response_gradient = (
                context.wp.empty(
                    (bands, inputs.spec.energies), dtype=context.wp.float64, device=context.device
                )
                if inputs.spec.active_response
                else None
            )
            response_flat = (
                self._response_gradient.flatten() if self._response_gradient is not None else None
            )
            self._response_gradient_rows = tuple(
                response_flat[b * inputs.spec.energies : (b + 1) * inputs.spec.energies]
                if response_flat is not None
                else None
                for b in range(bands)
            )
            cotangents_flat = self._cotangents.flatten()
            self._cotangent_rows = tuple(
                cotangents_flat[b * pixels : (b + 1) * pixels] for b in range(bands)
            )
        self._common_gradients = tuple(
            value
            for value in (self._paths_common, self._weights_common, self._coefficients_common)
            if value is not None
        )
        self._gradients = ResponseMatrixGradients(
            cast("wp.array[wp.float32] | None", self._fields_gradient),
            cast("wp.array[wp.float64] | None", self._pose_gradient),
            cast("wp.array[wp.float64] | None", self._coefficients_gradient),
            cast("FloatArray | None", self._weights_gradient),
            cast("wp.array[wp.float64] | None", self._response_gradient),
        )
        self._kernels = load_kernels("dpt.kernels.response_matrix")
        self._status_readback = StatusReadback(context, self._status)
        self._material = prepare_material_projection(
            grid,
            geometry,
            projection,
            materials=materials,
            fields=self._fields,
            out_paths=self._paths,
            cotangent_paths=adj_paths,
            out_fields=self._fields_gradient,
            device=context.device,
            stream=context.stream,
            field_domain=field_domain,
            channel_execution=channel_execution,
        )
        self._spectral = prepare_spectral(
            inputs.spec, max_pixels=pixels, device=context.device, stream=context.stream
        )
        self._checkpoint = StatusCheckpoint(
            context, (self._material, self._spectral, self), aggregate=False
        )

    def _common_gradient(self, destination: Any) -> _CommonGradient:
        context = self.context
        source64 = destination.dtype == context.wp.float64
        count = int(destination.size)
        return _CommonGradient(
            context.wp.empty_like(destination).flatten(),
            destination.flatten(),
            context.wp.zeros(
                count * (34 if source64 else 5), dtype=context.wp.uint64, device=context.device
            ),
            source64,
        )

    def check_status(self) -> None:
        """Check final cross-band cotangent representability."""
        raise_device_status(self._status_readback.read(), "response matrix")

    def clear_status(self) -> None:
        """Clear this owner's diagnostic after a composed checked boundary."""
        with self.context.scope():
            self._status.zero_()

    def consume_status(self) -> None:
        """Deliver and clear the exact accumulator's diagnostic word."""
        raise_device_status(self._status_readback.read(reset=True), "response matrix")

    @property
    def storage_bytes(self) -> int:
        """Owned factor, snapshot, image and derivative destination bytes."""
        return self._base_storage_bytes

    @property
    def scratch_bytes(self) -> int:
        """Device diagnostic, projection and native spectral reduction scratch."""
        return (
            self._material.scratch_bytes
            + self._spectral.scratch_bytes
            + self._aggregate_scratch_bytes
        )

    @property
    def host_bytes(self) -> int:
        """Persistent completion-readback staging bytes."""
        return (
            self._material.host_bytes + self._spectral.host_bytes + self._status_readback.host_bytes
        )

    @contextmanager
    def _evaluation(self) -> Generator[None, None, None]:
        require_no_tape()
        self.context.assert_stream(self.context.stream)
        if not self._lock.acquire(blocking=False):
            raise ContractError("a response matrix evaluation is already running")
        try:
            self._checkpoint.begin()
            yield
        except BaseException:
            if self.context.device.is_cuda:
                self.context.wp.synchronize_stream(self.context.stream)
            raise
        finally:
            self._lock.release()

    def forward(self, fields: wp.array[wp.float32], pose: wp.array[wp.float64]) -> FloatArray:
        """I[b,p] = sum_k w[k,p]*R[b,k]*exp(-sum_m mu[m,k]*A[m,p]) shares one material
        projection.

        Fields is FP32 (M,nz,ny,nx), (M,voxels) or flat material-major storage of
        dimensionless fractions/equivalent-basis fields. Pose is FP64 (12,), ordered
        row-major object-to-world rotation then translation in mm. ResponseMatrixInputs
        owns FP32 coefficients (M,K) in mm^-1, weights (K,) or (K,P) in integrated
        photon populations, and response (B,K) in signal per photon. The returned
        (B,P) array uses matched projection/spectral precision and contains linear
        means or their natural logarithms. Each successful call owns current field
        and pose copies for the selected active VJP parameters. Failed calls invalidate
        the saved state. Returned images belong to the pipeline and its next forward
        overwrites them. Reshape P to rows/columns for a planar detector.

        """
        with self._evaluation():
            self._has_forward = False
            context = self.context
            dimensions = getattr(fields, "ndim", 1)
            shape = (
                (self.inputs.spec.materials, *self.grid.shape)
                if dimensions == 4
                else (self.inputs.spec.materials, self.grid.voxels)
                if dimensions == 2
                else (self.inputs.spec.materials * self.grid.voxels,)
            )
            context.array(fields, "fields", dtype=context.wp.float32, shape=shape, ndim=len(shape))
            context.array(pose, "pose", dtype=context.wp.float64, shape=(12,))
            with context.scope():
                context.wp.copy(self._fields.flatten(), fields.flatten(), stream=context.stream)
                context.wp.copy(self._pose, pose, stream=context.stream)
            project_material_paths(self._pose, workspace=self._material, validate=True)
            for response, output in zip(self.inputs._response_rows, self._image_rows, strict=True):
                spectral_signal(
                    self._paths,
                    self.inputs.coefficients,
                    self.inputs.weights,
                    response,
                    out_mean=output,
                    workspace=self._spectral,
                    stream=context.stream,
                    validate=False,
                )
            self._checkpoint.consume()
            self._has_forward = True
            return self._images

    def vjp(self, grad_images: FloatArray) -> ResponseMatrixGradients:
        """Grad = sum_b J_b^T*grad_images[b,:] combines bands before one material reverse.

        Grad_images is (B,P) or contiguous (B*P,) in retained image precision.
        The latest successful forward owns FP32 fields (M,nz,ny,nx) and FP64 pose
        (12,). ResponseMatrixGradients returns FP32 fields (M,nz,ny,nx), FP64
        local-right pose (6,) in mm/rad, FP64 coefficients (M,K), FP64 shared
        weights (K,) or FP32 pixel weights (K,P), and FP64 responses (B,K).
        Fixed parameters return None. Each native band recomputes its first-order
        spectral products, then finite common path/coefficient/population contributions
        are summed exactly before one final storage rounding. The combined path
        covector drives one canonical material VJP. An individual unrepresentable
        band derivative remains an error. The caller's covector stays unchanged
        and the next VJP overwrites pipeline-owned gradient buffers.

        """
        with self._evaluation():
            if not self._has_forward:
                raise ContractError("run a successful forward before requesting its VJP")
            if not self._common_gradients and self._response_gradient is None:
                raise ContractError("request at least one active response matrix gradient")
            context = self.context
            shape = (
                (self.inputs.bands, self.inputs.pixels)
                if getattr(grad_images, "ndim", 1) == 2
                else (self.inputs.bands * self.inputs.pixels,)
            )
            context.array(
                grad_images, "grad_images", dtype=self._spectral.dtype, shape=shape, ndim=len(shape)
            )
            with context.scope():
                context.wp.copy(
                    self._cotangents.flatten(),
                    cast(Any, grad_images).flatten(),
                    stream=context.stream,
                )
                self._spectral._check_values([(self._cotangents.flatten(), False)])
                for common in self._common_gradients:
                    common.limbs.zero_()
                for band in range(self.inputs.bands):
                    spectral_vjp(
                        self._paths,
                        self.inputs.coefficients,
                        self.inputs.weights,
                        self.inputs._response_rows[band],
                        cotangent=self._cotangent_rows[band],
                        out_grad_paths=self._paths_common.source if self._paths_common else None,
                        out_grad_weights=self._weights_common.source
                        if self._weights_common
                        else None,
                        out_grad_coefficients=self._coefficients_common.source
                        if self._coefficients_common
                        else None,
                        out_grad_response=self._response_gradient_rows[band],
                        workspace=self._spectral,
                        stream=context.stream,
                        validate=False,
                    )
                    for common in self._common_gradients:
                        context.wp.launch(
                            self._kernels.get_accumulate(common.source64),
                            dim=common.source.size,
                            inputs=[common.source, common.source.size],
                            outputs=[common.limbs, self._status],
                            device=context.device,
                            stream=context.stream,
                            record_tape=False,
                        )
                for common in self._common_gradients:
                    context.wp.launch(
                        self._kernels.get_finish(common.source64, common.source64),
                        dim=common.destination.size,
                        inputs=[common.limbs, common.destination.size],
                        outputs=[common.destination, self._status],
                        device=context.device,
                        stream=context.stream,
                        record_tape=False,
                    )
            # Check every native band and final common cotangent before ray
            # traversal consumes the combined material seed.
            self._checkpoint.consume()
            if self._fields_gradient is not None or self._pose_gradient is not None:
                material_projection_vjp(
                    self._pose,
                    workspace=self._material,
                    out_pose_tangent=self._pose_gradient,
                    stream=context.stream,
                    validate=False,
                )
                self._checkpoint.consume()
            return self._gradients
