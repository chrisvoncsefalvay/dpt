"""Host PSF normalisation and amplification contracts."""

import math

import numpy as np
import pytest

from dpt.contracts import ContractError
from dpt.detector import BlurSpec
from dpt.materials import Provenance

FIXTURE = Provenance("analytic:Gaussian PSF", "0" * 64, "Apache-2.0", "Synthetic stencil")


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_normalised_gaussian_psfs_accept_sum_roundoff(dtype):
    straddled_one = 0
    for size in (3, 7, 15, 31):
        coordinate = np.arange(size, dtype=dtype) - size // 2
        for sigma in np.linspace(0.4, 6.0, 25):
            weights = np.exp(
                -(coordinate[:, None] ** 2 + coordinate[None, :] ** 2) / dtype(2 * sigma**2)
            ).astype(dtype)
            weights /= weights.sum()
            supplied = tuple(float(x) for x in weights.ravel())
            straddled_one += math.fsum(supplied) > 1
            spec = BlurSpec(35, 35, size, size, supplied, FIXTURE)
            assert spec.weights == supplied  # Never change a caller's PSF silently.
    # The former strict fsum > 1 condition spuriously rejected this fixture.
    assert straddled_one > 0


def test_blur_still_rejects_real_amplification_and_preserves_loss():
    with pytest.raises(ContractError, match="at most one"):
        BlurSpec(1, 1, 1, 1, (1.0001,), FIXTURE)
    assert BlurSpec(1, 1, 1, 1, (0.75,), FIXTURE).weights == (0.75,)
