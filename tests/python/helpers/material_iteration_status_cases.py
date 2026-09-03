"""Ordered resident diagnostics against the actual composed checkpoint law."""

from typing import Any

import numpy as np
import pytest

from dpt.contracts import ContractError, NumericalError
from dpt.material_iteration_graph import _decode_status_snapshot
from dpt.material_reconstruction import MaterialReconstruction
from helpers.material_inputs import _inputs

wp = pytest.importorskip("warp")
from dpt.kernels import material_iteration as controls  # noqa: E402


@pytest.mark.parametrize(
    "first,first_fixed,second,second_fixed,fatal,numerical",
    (
        (2, 1, 1, 1, 0, 1),
        (8, 1, 1, 1, 0, 1),
        (1, 1, 2, 1, 1, 0),
        (1, 0, 1, 1, 0, 1),
        (0, 1, 1, 1, 1, 0),
        (0, 1, 2, 1, 0, 1),
    ),
)
def test_trial_preserves_first_status(
    device, first, first_fixed, second, second_fixed, fatal, numerical
):
    flags = wp.zeros(controls.FLAG_SIZE, dtype=wp.int32, device=device)
    snapshots = wp.zeros(2, dtype=wp.int32, device=device)
    sources = [wp.array([code], dtype=wp.int32, device=device) for code in (first, second)]
    for index, (source, fixed) in enumerate(zip(sources, (first_fixed, second_fixed), strict=True)):
        wp.launch(
            controls.collect_trial_status,
            dim=1,
            inputs=[source, snapshots, 2, index, flags, fixed],
            device=device,
        )
    values = flags.numpy()
    assert values[controls.FATAL] == fatal
    assert values[controls.NUMERICAL] == numerical
    np.testing.assert_array_equal(snapshots.numpy(), [first, second])
    for source in sources:
        assert source.numpy()[0] == 0


@pytest.mark.parametrize("nested,own", ((2, 1), (1, 2), (0, 1), (2, 0)))
def test_snapshot_matches_nested_status(device, nested, own):
    supplied = _inputs()
    supplied["device"] = device
    solver = MaterialReconstruction(**supplied)
    material = solver.views[0].projection(True)
    checkpoint: Any = solver._evaluation_checkpoints[True]
    entries = checkpoint._entries
    codes = []
    with solver.context.scope():
        for owner, status in entries:
            code = nested if owner is material.projection else own if owner is material else 0
            status.assign(np.array([code], dtype=np.int32))
            codes.append(code)
        with pytest.raises((ContractError, NumericalError)) as captured:
            checkpoint._deliver(codes, consume=False)
        actual = _decode_status_snapshot(solver, entries, codes)
        expected = captured.value
        assert actual is not None
        assert type(actual) is type(expected)
        assert str(actual) == str(expected)
        assert actual.__notes__ == expected.__notes__
        assert actual.__dict__["status_diagnostics"] == expected.__dict__["status_diagnostics"]
        for (_, status), code in zip(entries, codes, strict=True):
            assert status.numpy()[0] == code
        with pytest.raises(type(expected)):
            checkpoint.consume()
        for _, status in entries:
            assert status.numpy()[0] == 0
