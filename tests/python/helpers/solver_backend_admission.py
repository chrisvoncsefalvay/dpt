"""Native backend primitives for independently bounded exercised solver inputs."""

# pyright: reportInvalidTypeForm=false, reportMissingImports=false
# pyright: reportUnknownMemberType=false, reportUnknownParameterType=false
# pyright: reportUntypedFunctionDecorator=false
# pyright: reportOperatorIssue=false
from functools import cache

import warp as wp

from dpt.kernels.spectral import get_attenuation, get_term_from_attenuation

wp.set_module_options({"fast_math": False, "fuse_fp": False, "enable_backward": False})


@wp.kernel
def primitive64(inputs: wp.array(dtype=wp.float64), output: wp.array(dtype=wp.float64)):
    i = wp.tid()
    output[2 * i] = wp.exp(-inputs[i])
    output[2 * i + 1] = wp.log(inputs[i])


@wp.kernel
def primitive32(inputs: wp.array(dtype=wp.float32), output: wp.array(dtype=wp.float32)):
    i = wp.tid()
    output[2 * i] = wp.exp(-inputs[i])
    output[2 * i + 1] = wp.log(inputs[i])


@cache
def get_vjp_attenuation_admission(wide: bool, arithmetic32: bool):
    attenuation = get_attenuation(wide, arithmetic32)
    term = get_term_from_attenuation(wide)

    @wp.kernel(module="unique")
    def capture(inputs: wp.array(dtype=wp.float64), output: wp.array(dtype=wp.float64)):
        i = wp.tid()
        state = attenuation(inputs[i])
        output[i] = term(
            state, wp.float64(8192.0), wp.float64(1.0), wp.float64(1.0), wp.float64(1.0)
        )

    return capture
