"""Prepared FP64 tile trees shared by deterministic scalar and pose reductions.

Only preparation allocates. The first level may be filled by an operator's
fused contribution kernel, or the tree may consume an existing FP64 vector.
Logical counts bound every read from reusable capacity-sized scratch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from dpt._runtime import DeviceContext, load_kernels


@dataclass(frozen=True, slots=True)
class ReductionTree:
    context: DeviceContext
    tile: int
    partials: tuple[Any, ...]
    _kernels: Any = field(repr=False)

    @property
    def scratch_bytes(self) -> int:
        return sum(int(array.size) * 8 for array in self.partials)

    def finish(
        self,
        count: int,
        destination: Any,
        status: Any,
        *,
        components: int = 1,
        source: Any = None,
        accumulate: bool = False,
    ) -> None:
        """Reduce logical rows and store once, flagging nonfinite final values.

        With no source, count denotes rows already written to partials[0].
        Otherwise source contains count packed rows of FP64 components.
        Empty sums write zero without reading uninitialised scratch.
        """
        ctx, wp = self.context, self.context.wp
        nonempty = count > 0
        previous = self.partials[0] if source is None else source
        destinations = self.partials[1:] if source is None else self.partials
        for following in destinations:
            if count <= 1:
                break
            next_count = (count + self.tile - 1) // self.tile
            wp.launch_tiled(
                self._kernels.reduce_kernel(self.tile, components),
                dim=next_count,
                inputs=[previous, count],
                outputs=[following],
                block_dim=self.tile,
                device=ctx.device,
                stream=ctx.stream,
                record_tape=False,
            )
            previous, count = following, next_count
        wp.launch(
            self._kernels.finish_kernel(destination.dtype == wp.float32, accumulate),
            dim=components,
            inputs=[previous, nonempty],
            outputs=[destination, status],
            device=ctx.device,
            stream=ctx.stream,
            record_tape=False,
        )


def prepare_reduction(
    context: DeviceContext, max_values: int, *, tile: int = 256, components: int = 1
) -> ReductionTree:
    """Allocate a complete tree for at most max_values input rows."""
    wp = context.wp
    count = max(1, (max_values + tile - 1) // tile if context.device.is_cuda else max_values)
    partials: list[Any] = []
    with context.scope():
        while True:
            partials.append(wp.empty(count * components, dtype=wp.float64, device=context.device))
            if count == 1:
                break
            count = (count + tile - 1) // tile
    return ReductionTree(context, tile, tuple(partials), load_kernels("dpt.kernels.reductions"))
