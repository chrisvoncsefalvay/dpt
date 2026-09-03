"""Native bounded auxiliary draws for the complete geometry estimator design."""

# Warp annotations are runtime DSL types.
# pyright: reportInvalidTypeForm=false, reportUnknownParameterType=false
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false
# pyright: reportUnknownVariableType=false, reportUntypedFunctionDecorator=false
# pyright: reportMissingImports=false, reportUntypedClassDecorator=false
# Struct annotations, scalar bitwise operations and vector indexing are compiled
# by Warp's DSL rather than evaluated with the provider's Python stub classes.
# pyright: reportGeneralTypeIssues=false, reportOperatorIssue=false
# pyright: reportReturnType=false, reportIndexIssue=false, reportArgumentType=false
import warp as wp

from dpt.kernels.random import random4

from .sampling_kernels import RandomDesign, design_uniform, random_dimension, transport_random4


@wp.struct
class GeometryRandomDesign:
    physical: RandomDesign
    event_guard: int
    physical_dimensions: int
    dimensions: int


@wp.func
def geometry_auxiliary_dimension(
    event: wp.uint32, domain: wp.uint32, lane: int, design: GeometryRandomDesign
) -> int:
    """Return an absolute column, -1 for an unused lane or -2 for invalid input."""
    dimension = wp.int32(-2)
    if event < wp.uint32(design.event_guard):
        role = domain ^ wp.uint32(0x30000000)
        if role < wp.uint32(3):
            if role == wp.uint32(0):
                dimension = design.physical_dimensions + 10 * int(event) + lane
            else:
                dimension = wp.int32(-1)
                if lane < 3:
                    offset = wp.int32(4)
                    if role == wp.uint32(2):
                        offset = wp.int32(7)
                    dimension = design.physical_dimensions + 10 * int(event) + offset + lane
        elif event == wp.uint32(0) and role == wp.uint32(3):
            dimension = wp.int32(-1)
            if lane < 3:
                dimension = design.physical_dimensions + 10 * design.event_guard + lane
        elif event == wp.uint32(0) and role == wp.uint32(4):
            dimension = wp.int32(-1)
            if lane < 2:
                dimension = design.physical_dimensions + 10 * design.event_guard + 3 + lane
    return dimension


@wp.func
def geometry_auxiliary_random4(
    seed: wp.uint64,
    history: wp.uint64,
    event: wp.uint32,
    domain: wp.uint32,
    design: GeometryRandomDesign,
) -> wp.vec4d:
    """Evaluate geometry roles with no IID fallback or physical column reuse."""
    draws = wp.vec4d(wp.float64(wp.nan))
    physical = design.physical
    if (
        seed == physical.seed
        and history >= physical.first_history
        and history - physical.first_history < wp.uint64(physical.count)
    ):
        jitter = wp.vec4d(wp.float64(0.5))
        if physical.kind == 0 or physical.kind == 1:
            jitter = random4(seed, history, event, domain)
        index = wp.uint32(history - physical.first_history)
        for lane in range(4):
            dimension = geometry_auxiliary_dimension(event, domain, lane, design)
            if dimension >= design.dimensions:
                draws[lane] = wp.float64(wp.nan)
            elif physical.kind == 0:
                if dimension >= 0:
                    draws[lane] = jitter[lane]
                elif dimension == -1:
                    draws[lane] = wp.float64(0.5)
            else:
                draws[lane] = design_uniform(index, dimension, jitter[lane], physical)
    return draws


@wp.func
def geometry_physical_random4(
    seed: wp.uint64,
    history: wp.uint64,
    branch: int,
    event_offset: wp.uint32,
    event: wp.uint32,
    domain: wp.uint32,
    design: GeometryRandomDesign,
) -> wp.vec4d:
    """Preserve canonical physical draws and enforce local continuation guards."""
    draws = wp.vec4d(wp.float64(wp.nan))
    source = event == wp.uint32(0x80000000) or event == wp.uint32(0x80000002)
    valid = False
    if source:
        valid = branch == 0 and event_offset == wp.uint32(0) and domain == wp.uint32(0)
    else:
        valid = (
            branch >= 0
            and branch < design.physical.max_branches
            and event < wp.uint32(design.event_guard)
            and event_offset < wp.uint32(design.physical.max_events)
            and event < wp.uint32(design.physical.max_events) - event_offset
            and (domain & wp.uint32(0x0FFF0000)) == wp.uint32(0)
            and (domain & wp.uint32(0x30000000)) != wp.uint32(0x30000000)
            and (branch >= 3 or event_offset == wp.uint32(0))
            and event_offset % wp.uint32(design.event_guard) == wp.uint32(0)
        )
        if valid:
            valid = (
                random_dimension(
                    event_offset + event, domain | wp.uint32(branch << 16), 0, design.physical
                )
                >= 0
            )
    valid = (
        valid
        and seed == design.physical.seed
        and history >= design.physical.first_history
        and history - design.physical.first_history < wp.uint64(design.physical.count)
    )
    if valid:
        draws = transport_random4(
            seed, history, event_offset + event, domain | wp.uint32(branch << 16), design.physical
        )
    return draws


@wp.kernel
def inspect_geometry_auxiliary(
    seed: wp.uint64,
    first_history: wp.uint64,
    event: wp.uint32,
    domain: wp.uint32,
    design: GeometryRandomDesign,
    out: wp.array(dtype=wp.vec4d),
):
    """Evaluate all four lanes for independent CPU/CUDA reference checks."""
    index = wp.tid()
    out[index] = geometry_auxiliary_random4(
        seed, first_history + wp.uint64(index), event, domain, design
    )


@wp.kernel
def inspect_geometry_physical(
    seed: wp.uint64,
    first_history: wp.uint64,
    branch: int,
    event_offset: wp.uint32,
    event: wp.uint32,
    domain: wp.uint32,
    design: GeometryRandomDesign,
    out: wp.array(dtype=wp.vec4d),
):
    """Inspect original lineage and conditional-continuation physical counters."""
    index = wp.tid()
    out[index] = geometry_physical_random4(
        seed, first_history + wp.uint64(index), branch, event_offset, event, domain, design
    )
