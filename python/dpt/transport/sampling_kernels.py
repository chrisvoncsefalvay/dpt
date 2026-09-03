"""Counter-compatible bounded full-path randomised design evaluation in Warp."""

# Warp annotations are runtime DSL types.
# pyright: reportInvalidTypeForm=false, reportUnknownParameterType=false
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false
# pyright: reportUnknownVariableType=false, reportUntypedFunctionDecorator=false
# pyright: reportMissingImports=false, reportUntypedClassDecorator=false
import warp as wp

from dpt.kernels.random import random4


@wp.struct
class RandomDesign:
    kind: int
    seed: wp.uint64
    first_history: wp.uint64
    count: wp.uint32
    max_events: int
    max_trials: int
    max_branches: int
    layout: int
    stride: int
    dimensions: int
    words: wp.array(dtype=wp.uint32)
    shift: wp.array(dtype=wp.uint32)


@wp.func
def random_dimension(event: wp.uint32, domain: wp.uint32, lane: int, design: RandomDesign) -> int:
    """Return active dimension, -1 for an unused lane or -2 for an invalid address."""
    dimension = wp.int32(-2)
    if event == wp.uint32(0x80000000) and domain == wp.uint32(0):
        dimension = wp.int32(-1)
        if lane < 2:
            dimension = lane
    elif event == wp.uint32(0x80000002) and domain == wp.uint32(0):
        dimension = wp.int32(-1)
        if lane == 0:
            dimension = 2
    else:
        branch = int((domain >> wp.uint32(16)) & wp.uint32(0xFFF))
        role = int(domain & wp.uint32(0xFFFF))
        namespace = int(domain >> wp.uint32(30))
        extension = int((domain >> wp.uint32(28)) & wp.uint32(3))
        if (
            design.layout == 2
            and namespace == 0
            and extension > 0
            and extension < 3
            and role == 0
            and event < wp.uint32(design.max_events)
            and branch < design.max_branches
        ):
            dimension = wp.int32(-1)
            if lane < 2:
                base = 3 + (branch * design.max_events + int(event)) * design.stride
                dimension = base + 12 + 6 * design.max_trials + 2 * (extension - 1) + lane
        if (
            event < wp.uint32(design.max_events)
            and branch < design.max_branches
            and (domain & wp.uint32(0x30000000)) == wp.uint32(0)
        ):
            base = 3 + (branch * design.max_events + int(event)) * design.stride
            if namespace == 0:
                if role == 0:
                    dimension = base + lane
                elif role == 1:
                    dimension = wp.int32(-1)
                    if lane == 0:
                        dimension = base + 4
                elif role == 2:
                    dimension = wp.int32(-1)
                    if lane < 3:
                        dimension = base + 5 + lane
                elif role == 3:
                    dimension = wp.int32(-1)
                    if lane == 0:
                        dimension = base + 8
            elif namespace == 1:
                if role == 0:
                    dimension = wp.int32(-1)
                    if lane < 2:
                        dimension = base + 9 + lane
                elif role == 1:
                    dimension = wp.int32(-1)
                    if lane == 0:
                        dimension = base + 11
            elif design.layout == 1:
                blocks = (design.max_trials + 1) // 2
                if namespace == 2 and role < blocks:
                    dimension = base + 12 + 4 * role + lane
                elif namespace == 3 and role < design.max_trials:
                    dimension = wp.int32(-1)
                    if lane < 3:
                        dimension = base + 12 + 4 * blocks + 3 * role + lane
            elif role < design.max_trials:
                dimension = wp.int32(-1)
                if lane < 3:
                    dimension = (
                        base + 12 + (namespace - 2) * 3 * design.max_trials + 3 * role + lane
                    )
    return dimension


@wp.func
def design_uniform(
    index: wp.uint32, dimension: int, jitter: wp.float64, design: RandomDesign
) -> wp.float64:
    value = wp.float64(0.5)
    if dimension == -2:
        value = wp.float64(wp.nan)
    elif dimension >= 0:
        if design.kind == 1:
            stratum = (index * design.words[dimension] + design.shift[dimension]) & (
                design.count - wp.uint32(1)
            )
            value = wp.min(
                (wp.float64(stratum) + jitter) / wp.float64(design.count),
                wp.float64(0.9999999999999999),
            )
        elif design.kind == 2:
            gray = index ^ (index >> wp.uint32(1))
            word = design.shift[dimension]
            bit = wp.int32(0)
            while gray != wp.uint32(0):
                if (gray & wp.uint32(1)) != wp.uint32(0):
                    word = word ^ design.words[dimension * 32 + bit]
                gray = gray >> wp.uint32(1)
                bit += 1
            value = (wp.float64(word) + wp.float64(0.5)) * wp.float64(1.0 / 4294967296.0)
        else:
            value = wp.float64(wp.nan)
    return value


@wp.func
def transport_random4(
    seed: wp.uint64, history: wp.uint64, event: wp.uint32, domain: wp.uint32, design: RandomDesign
) -> wp.vec4d:
    """Evaluate the declared law with original history, event, domain and lane.

    Kind zero delegates directly to the historical Philox implementation. Every
    other kind uses a bounded resident design and produces NaNs for invalid
    identities or counters, which invalidate the transport estimate. Unused
    lanes are exactly 0.5 and never supply a random role.
    """
    draws = wp.vec4d()
    if design.kind == 0:
        draws = random4(seed, history, event, domain)
    elif (
        seed != design.seed
        or history < design.first_history
        or (history - design.first_history >= wp.uint64(design.count))
    ):
        draws = wp.vec4d(wp.float64(wp.nan))
    else:
        jitter = wp.vec4d(wp.float64(0.5))
        if design.kind == 1:
            jitter = random4(seed, history, event, domain)
        index = wp.uint32(history - design.first_history)
        for lane in range(4):
            dimension = random_dimension(event, domain, lane, design)
            draws[lane] = design_uniform(index, dimension, jitter[lane], design)
    return draws


@wp.kernel
def inspect_random_design(
    seed: wp.uint64,
    first_history: wp.uint64,
    event: wp.uint32,
    domain: wp.uint32,
    design: RandomDesign,
    out: wp.array(dtype=wp.vec4d),
):
    """Evaluate declared coordinates for independent CPU/CUDA oracle checks."""
    index = wp.tid()
    out[index] = transport_random4(seed, first_history + wp.uint64(index), event, domain, design)


@wp.kernel
def sample_parallel_source(
    lower: wp.vec3d,
    extent: wp.vec2d,
    direction: wp.vec3d,
    weight: wp.float64,
    seed: wp.uint64,
    first_history: wp.uint64,
    out_position: wp.array(dtype=wp.vec3d),
    out_direction: wp.array(dtype=wp.vec3d),
    out_weight: wp.array(dtype=wp.float64),
    design: RandomDesign,
):
    index = wp.tid()
    draw = transport_random4(
        seed, first_history + wp.uint64(index), wp.uint32(0x80000000), wp.uint32(0), design
    )
    out_position[index] = lower + wp.vec3d(
        extent[0] * draw[0], extent[1] * draw[1], wp.float64(0.0)
    )
    out_direction[index] = direction
    out_weight[index] = weight


@wp.kernel
def sample_point_source(
    source: wp.vec3d,
    corner: wp.vec3d,
    u: wp.vec3d,
    v: wp.vec3d,
    extent: wp.vec2d,
    weight: wp.float64,
    seed: wp.uint64,
    first_history: wp.uint64,
    out_position: wp.array(dtype=wp.vec3d),
    out_direction: wp.array(dtype=wp.vec3d),
    out_weight: wp.array(dtype=wp.float64),
    design: RandomDesign,
):
    index = wp.tid()
    draw = transport_random4(
        seed, first_history + wp.uint64(index), wp.uint32(0x80000000), wp.uint32(0), design
    )
    endpoint = corner + extent[0] * draw[0] * u + extent[1] * draw[1] * v
    out_position[index] = source
    delta = endpoint - source
    scale = wp.max(wp.abs(delta[0]), wp.max(wp.abs(delta[1]), wp.abs(delta[2])))
    out_direction[index] = wp.normalize(delta / scale)
    out_weight[index] = weight
