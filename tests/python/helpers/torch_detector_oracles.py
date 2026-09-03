"""Independent detector matrix and objective references, without Torch or CUDA imports."""

import math

import numpy as np

from dpt.detector import SeparableBlurSpec


def blur_matrix(spec):
    height, width = spec.height, spec.width
    if isinstance(spec, SeparableBlurSpec):
        stencil = np.outer(spec.vertical, spec.horizontal)
    else:
        stencil = np.asarray(spec.weights).reshape(spec.kernel_height, spec.kernel_width)
    matrix = np.zeros((height * width, height * width))
    for row in range(height):
        for column in range(width):
            for i in range(stencil.shape[0]):
                for j in range(stencil.shape[1]):
                    source_row = row + i - stencil.shape[0] // 2
                    source_column = column + j - stencil.shape[1] // 2
                    if 0 <= source_row < height and 0 <= source_column < width:
                        matrix[row * width + column, source_row * width + source_column] += stencil[
                            i, j
                        ]
    return matrix


def objective_reference(prediction, observation, weights, valid, kind, reduction):
    values = np.zeros(prediction.size)
    gradients = np.zeros(prediction.size)
    for pixel, (p, y, w, mask) in enumerate(
        zip(
            prediction.reshape(-1),
            observation.reshape(-1),
            weights.reshape(-1),
            valid.reshape(-1),
            strict=True,
        )
    ):
        if mask == 0:
            continue
        if kind == "squared_error":
            values[pixel], gradients[pixel] = 0.5 * w * (p - y) ** 2, w * (p - y)
        elif y == 0:
            values[pixel], gradients[pixel] = w * p, w
        else:
            values[pixel] = w * (p - y + y * math.log(y / p))
            gradients[pixel] = w * (1 - y / p)
    divisor = prediction.size if reduction == "mean" else 1
    return values.sum() / divisor, gradients.reshape(prediction.shape) / divisor
