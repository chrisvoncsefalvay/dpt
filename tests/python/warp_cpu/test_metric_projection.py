"""Exact rational simplex QP oracle, independent of the KKT implementation."""

import importlib
import random
from typing import Any

import numpy as np

from dpt.kernels.material_reconstruction import projected_metric_step
from helpers.metric_oracles import exact_qp

wp: Any = importlib.import_module("warp")


def test_metric_projection_matches_exact_qp_through_condition_one_million():
    rng = random.Random(381)
    for metric in (
        (1.0, 0.0, 100.0),
        (1.0, 0.0, 1e6),
        (1e6, 0.0, 1.0),
        (2.0, 0.3, 1.0),
        (1.0, -0.3, 2.0),
    ):
        for i in range(28):
            field = np.asarray((rng.random() * 0.4, rng.random() * 0.4), dtype=np.float32)
            gradient = np.asarray((rng.uniform(-30, 30), rng.uniform(-30, 30)), dtype=np.float32)
            if i == 0:
                gradient[:] = (-1e5, 1e5)
            if i == 1:
                gradient[:] = (1e5, -1e5)
            if i == 2:
                gradient[:] = (-1e5, -1e5)
            scale = np.float32(0.7 if i % 2 else 1.3)
            step = 0.13 if i % 3 else 0.37
            a, b, c = metric
            det = a * c - b * b
            f = wp.array(field, dtype=wp.float32, device="cpu")
            g = wp.array(gradient, dtype=wp.float32, device="cpu")
            out = wp.zeros(2, dtype=wp.float32, device="cpu")
            slopes = wp.zeros(2, dtype=wp.float64, device="cpu")
            mapping = wp.zeros(1, dtype=wp.float64, device="cpu")
            status = wp.zeros(1, dtype=wp.int32, device="cpu")
            scales = wp.array([scale], dtype=wp.float32, device="cpu")
            wp.launch(
                projected_metric_step,
                dim=1,
                inputs=[
                    f,
                    g,
                    1,
                    step,
                    f,
                    0.0,
                    True,
                    scales,
                    wp.vec3d(*metric),
                    wp.vec3d(c / det, -b / det, a / det),
                    out,
                    slopes,
                    mapping,
                    status,
                ],
                device="cpu",
            )
            assert status.numpy()[0] == 0, (metric, i)
            expected = np.asarray(
                [float(v) for v in exact_qp(field, gradient, step / float(scale), metric)]
            )
            np.testing.assert_allclose(out.numpy(), expected, rtol=0, atol=2e-7)
