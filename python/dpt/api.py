"""The direct real-volume to device-projection path."""

from __future__ import annotations

from typing import TYPE_CHECKING

from dpt.contracts import ContractError
from dpt.imaging_io import ImagingVolume

if TYPE_CHECKING:
    import torch

    from dpt.torch.modules import GeometryInput, Precision, Projector

__all__ = ["project"]


def project(
    volume: ImagingVolume | torch.Tensor,
    geometry: GeometryInput | None = None,
    *,
    device: str | torch.device | None = None,
    precision: Precision | None = None,
    samples_per_ray: int | None = None,
    step_mm: float | None = None,
    pose: torch.Tensor | None = None,
    packed_pose: torch.Tensor | None = None,
    prepared: Projector | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return device optical depth ``(views,rows,columns)`` from attenuation.

    ``volume`` normally holds immutable mm^-1 values and physical geometry.
    Upload a single owned FP32 field and use the same ``dpt.torch.Projector``
    as the differentiable workflow. New calls default to CUDA, FP32 and
    1024 midpoint samples. ``step_mm`` resolves a fixed count at preparation.
    No projection is copied to NumPy or downloaded for display here.

    Pass ``prepared=Projector(...)`` to reuse geometry and device workspaces;
    omitted precision/sampling settings then come from that module. A device
    attenuation tensor may also be passed with ``prepared`` and retains its
    autograd graph without casting or copying. ``pose`` denotes six right
    tangent coordinates; ``packed_pose`` denotes twelve transform entries.
    ``out`` is an inference-only buffer; omit it for differentiable results.
    """
    import torch

    from dpt.torch import Projector

    if prepared is not None and not isinstance(prepared, Projector):
        raise ContractError("prepared must be a dpt.torch.Projector")
    if isinstance(volume, ImagingVolume):
        if volume.metadata.value_units != "mm^-1":
            raise ContractError("projection requires attenuation units='mm^-1'; convert HU first")
        grid = volume.grid
        target = "cuda" if device is None else device
        field = torch.tensor(volume.values, dtype=torch.float32, device=target)
    elif isinstance(volume, torch.Tensor):
        if prepared is None:
            raise ContractError(
                "a device tensor requires prepared=Projector(grid=..., geometry=...)"
            )
        grid = prepared.grid
        field = volume
        if device is not None:
            requested = torch.device(device)
            same_device = requested == field.device or (
                requested.type == field.device.type and requested.index is None
            )
            if not same_device:
                raise ContractError("device must match the supplied tensor; move it explicitly")
    else:
        raise ContractError("volume must be an ImagingVolume or a prepared device tensor")
    if prepared is None:
        if geometry is None:
            raise ContractError("geometry is required when no prepared Projector is supplied")
        prepared = Projector(
            grid=grid,
            geometry=geometry,
            precision="fp32" if precision is None else precision,
            samples_per_ray=samples_per_ray,
            step_mm=step_mm,
        )
    else:
        if grid != prepared.grid:
            raise ContractError("volume grid does not match the prepared Projector")
        if geometry is not None:
            from dpt.torch.modules import _geometry

            if _geometry(geometry) != prepared.geometry:
                raise ContractError("geometry does not match the prepared Projector")
        if precision is not None and precision != prepared.precision:
            raise ContractError("precision does not match the prepared Projector")
        if samples_per_ray is not None and samples_per_ray != prepared.samples_per_ray:
            raise ContractError("samples_per_ray does not match the prepared Projector")
        if step_mm is not None and step_mm != prepared.step_mm:
            raise ContractError("step_mm does not match the prepared Projector")
    return prepared(field, pose, packed_pose=packed_pose, out=out)
