"""Compose rigid poses as ``G(xi) = anchor @ exp(hat(xi))`` in Torch.

Packed poses use twelve FP64 values: row-major rotation and translation
in mm. Six-coordinate increments use local translation in mm followed
by rotation in radians. Torch differentiates the SE(3) exponential and
carries its translation/rotation coupling into packed-pose operators.

"""

from __future__ import annotations

# The optional framework loader is shared with the public interop adapter.
# pyright: reportPrivateUsage=false
from typing import Any

from dpt.contracts import ContractError


def compose_pose(anchor: Any, tangent: Any) -> Any:
    """Compute ``G(xi) = anchor @ exp(hat(xi))`` with a differentiable SE(3) chart.

    ``anchor`` has trailing shape ``(12,)``: row-major rotation followed
    by translation in mm. ``tangent`` has trailing shape ``(6,)`` and
    order ``(tx,ty,tz,rx,ry,rz)`` in mm and radians. Leading dimensions
    broadcast. Both tensors use FP64 on one CPU or CUDA device; the
    contiguous result has the broadcast leading shape followed by 12.

    The increment acts on the right, in object axes. At zero tangent,
    its derivative is the projector's local-right chart. At nonzero
    rotation, the SE(3) exponential's left Jacobian couples translation
    and rotation. Torch carries gradients to both anchor and tangent
    when their ``requires_grad`` flags are set.

    """
    import torch

    if not isinstance(anchor, torch.Tensor) or not isinstance(tangent, torch.Tensor):
        raise ContractError("anchor and tangent must be Torch tensors")
    if anchor.dtype != torch.float64 or tangent.dtype != torch.float64:
        raise ContractError("rigid pose charts require float64 tensors")
    if anchor.device != tangent.device or anchor.device.type not in ("cpu", "cuda"):
        raise ContractError("anchor and tangent must share a CPU or CUDA device")
    if anchor.ndim < 1 or anchor.shape[-1] != 12 or tangent.ndim < 1 or tangent.shape[-1] != 6:
        raise ContractError("anchor and tangent require trailing shapes (12,) and (6,)")
    if any(count == 0 for count in (*anchor.shape, *tangent.shape)):
        raise ContractError("pose charts require nonempty tensor dimensions")
    try:
        shape = torch.broadcast_shapes(anchor.shape[:-1], tangent.shape[:-1])
    except RuntimeError as error:
        raise ContractError("anchor and tangent leading dimensions do not broadcast") from error
    translation, omega = tangent[..., :3], tangent[..., 3:]
    x, y, z = omega.unbind(-1)
    zero = torch.zeros_like(x)
    skew = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), dim=-1).reshape(
        *tangent.shape[:-1], 3, 3
    )
    top = torch.cat((skew, translation.unsqueeze(-1)), dim=-1)
    bottom = torch.zeros((*tangent.shape[:-1], 1, 4), dtype=tangent.dtype, device=tangent.device)
    increment = torch.linalg.matrix_exp(torch.cat((top, bottom), dim=-2))
    rotation = anchor[..., :9].reshape(*anchor.shape[:-1], 3, 3)
    shifted = (rotation @ increment[..., :3, 3].unsqueeze(-1)).squeeze(-1) + anchor[..., 9:]
    composed = rotation @ increment[..., :3, :3]
    return torch.cat((composed.reshape(*shape, 9), shifted), dim=-1).contiguous()
