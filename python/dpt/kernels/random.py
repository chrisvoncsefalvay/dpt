"""Counter-addressed Philox4x32-10 shared by transport and observations.

The counter is (identity low, identity high, event, domain); the key is the
64-bit seed. All coordinates are explicit: scheduling and rejection in another
history cannot change this history's stream. Domain 0 belongs to transport.
Domains 1..0x7fffffff belong to observations. The paired electronic Gaussian
mode reserves domain 0xffffffff. Transport rejection uses the 0x8 and 0xc
namespaces with 12-bit branch and 16-bit proposal coordinates. Counter
coordinates must never wrap; the host boundary rejects exhausted identities.

Constants and round schedule: Random123, Salmon et al. (SC11), Philox4x32-10,
https://github.com/DEShawResearch/random123/blob/main/include/Random123/philox.h.
This is the published algorithm expressed in Warp, not a bundled copy of a
third-party implementation. These functions return no persistent RNG state.
"""

# Warp annotations are executable DSL expressions; host interfaces remain strict.
# The optional GPU import is resolved only when an operator is prepared.
# pyright: reportInvalidTypeForm=false, reportUnknownParameterType=false
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false
# pyright: reportUnknownVariableType=false, reportUntypedFunctionDecorator=false
# pyright: reportMissingImports=false, reportUntypedClassDecorator=false

import warp as wp


@wp.func
def random4(
    seed: wp.uint64,
    identity: wp.uint64,
    event: wp.uint32,
    domain: wp.uint32,
) -> wp.vec4d:
    """Return four open-interval uniforms; each has 32 random mantissa bits."""
    c0 = wp.uint32(identity & wp.uint64(0xFFFFFFFF))
    c1 = wp.uint32(identity >> wp.uint64(32))
    c2 = event
    c3 = domain
    k0 = wp.uint32(seed & wp.uint64(0xFFFFFFFF))
    k1 = wp.uint32(seed >> wp.uint64(32))
    for _ in range(10):
        product0 = wp.uint64(0xD2511F53) * wp.uint64(c0)
        product1 = wp.uint64(0xCD9E8D57) * wp.uint64(c2)
        low0 = wp.uint32(product0 & wp.uint64(0xFFFFFFFF))
        low1 = wp.uint32(product1 & wp.uint64(0xFFFFFFFF))
        high0 = wp.uint32(product0 >> wp.uint64(32))
        high1 = wp.uint32(product1 >> wp.uint64(32))
        c0 = high1 ^ c1 ^ k0
        c1 = low1
        c2 = high0 ^ c3 ^ k1
        c3 = low0
        k0 = k0 + wp.uint32(0x9E3779B9)
        k1 = k1 + wp.uint32(0xBB67AE85)
    scale = wp.float64(1.0 / 4294967296.0)
    return wp.vec4d(
        (wp.float64(c0) + wp.float64(0.5)) * scale,
        (wp.float64(c1) + wp.float64(0.5)) * scale,
        (wp.float64(c2) + wp.float64(0.5)) * scale,
        (wp.float64(c3) + wp.float64(0.5)) * scale,
    )
