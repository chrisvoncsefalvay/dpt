"""Shared exact scientific helpers for the native controls."""

# pyright: reportInvalidTypeForm=false, reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false
# pyright: reportUnknownParameterType=false, reportMissingParameterType=false
# pyright: reportUntypedFunctionDecorator=false

import math

import numpy as np


def independent_isotropic_cube(albedo: float, count: int):
    """All-orders homogeneous-cube analogue MC using NumPy's separate RNG and geometry."""
    rng = np.random.default_rng(319753)
    positions = np.tile([0.0, 0.0, 0.0], (count, 1))
    directions = np.tile([0.0, 0.0, 1.0], (count, 1))
    active = np.arange(count)
    pixels = np.full(count, -1, dtype=np.int32)
    lower, upper = np.array([-1.0, -1.0, 0.0]), np.array([1.0, 1.0, 2.0])
    for _ in range(128):
        if not len(active):
            break
        x, d = positions[active], directions[active]
        faces = np.where(d > 0, upper, lower)
        distances = np.full_like(d, np.inf)
        np.divide(faces - x, d, out=distances, where=d != 0)
        exit_distance = np.min(distances, axis=1)
        flight = rng.exponential(1 / 1.2, len(active))
        collision = flight < exit_distance
        escaped = active[~collision]
        x_exit = x[~collision] + exit_distance[~collision, None] * d[~collision]
        d_exit = d[~collision]
        toward_detector = d_exit[:, 2] > 0
        escaped = escaped[toward_detector]
        x_exit, d_exit = x_exit[toward_detector], d_exit[toward_detector]
        intersection = x_exit + ((3 - x_exit[:, 2]) / d_exit[:, 2])[:, None] * d_exit
        columns = np.floor((intersection[:, 0] + 3) / 2).astype(int)
        rows = np.floor((intersection[:, 1] + 4) / 4).astype(int)
        hit = (columns >= 0) & (columns < 3) & (rows >= 0) & (rows < 2)
        pixels[escaped[hit]] = rows[hit] * 3 + columns[hit]
        scattered = collision.copy()
        scattered[collision] = rng.random(np.sum(collision)) < albedo
        active = active[scattered]
        positions[active] = x[scattered] + flight[scattered, None] * d[scattered]
        cosine = rng.uniform(-1, 1, len(active))
        phi = rng.uniform(0, 2 * math.pi, len(active))
        sine = np.sqrt(1 - cosine**2)
        # An isotropic direction is uniform in a fixed frame too, independently
        # of the incoming ray; this avoids sharing production's rotation basis.
        directions[active] = np.column_stack((sine * np.cos(phi), sine * np.sin(phi), cosine))
    assert not len(active), "independent oracle guard exhausted"
    return np.bincount(pixels[pixels >= 0], minlength=6) / count
