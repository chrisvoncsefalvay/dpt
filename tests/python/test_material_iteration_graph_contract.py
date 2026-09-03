"""Cold preparation bounds for complete material iteration graphs."""

from types import SimpleNamespace
from typing import Any

import pytest

from dpt import material_iteration_graph as graphs
from dpt.material_iteration_graph import _reference_capacity


@pytest.mark.parametrize(
    "window,iterations,start,count,expected",
    (
        (4, 6, 0, 0, 4),
        (100, 2, 0, 0, 2),
        (100, 2, 1, 100, 100),
        (100, 2, 1, 17, 18),
        (4, 6, 4, 4, 4),
        (4, 6, 6, 4, 4),
        (4, 6, 6, 0, 1),
    ),
)
def test_gll_capacity_preserves_all_admitted_references(window, iterations, start, count, expected):
    capacity = _reference_capacity(window, iterations, start, count)
    assert capacity == expected
    assert capacity >= min(window, count)
    assert capacity >= min(window, count + iterations - start)


@pytest.mark.parametrize("stage", ("constructor", "replacement", "reset"))
@pytest.mark.parametrize("error_type", (RuntimeError, KeyboardInterrupt))
def test_setup_failure_discards_cache_and_preserves_accepted_state(monkeypatch, stage, error_type):
    """Injected boundary exceptions exercise no numerical graph implementation."""
    error = error_type("interrupted graph preparation")
    fields = object()
    history = [{"accepted": True}]
    callbacks = []
    state = SimpleNamespace(
        policy=SimpleNamespace(nonmonotone_window=4, iterations=1),
        start_iteration=0,
        reference_losses=[],
        history=history,
    )

    class Cached:
        window = 1
        binding = ("prepared",)

        def _binding(self):
            return ("changed",) if stage == "replacement" else self.binding

        def reset(self, state):
            raise error

    def interrupted(solver, state):
        raise error

    monkeypatch.setattr(graphs, "_IterationGraph", interrupted)
    solver: Any = SimpleNamespace(
        _iteration_graph_state=None if stage == "constructor" else Cached(),
        _accepted_forward_available=True,
        _prediction_state="accepted",
        fractions=fields,
    )
    with pytest.raises(error_type) as captured:
        graphs.iterate_material_graph(
            solver, state, None, None, callbacks.append, callbacks.append, None
        )
    assert captured.value is error
    assert solver._iteration_graph_state is None
    assert solver._accepted_forward_available is False
    assert solver._prediction_state == "uninitialised"
    assert solver.fractions is fields
    assert state.history is history
    assert state.history == [{"accepted": True}]
    assert callbacks == []
