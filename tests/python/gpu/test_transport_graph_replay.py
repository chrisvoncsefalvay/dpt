"""Actual CUDA capture/replay of explicitly fixed original-history identities."""

import importlib
import math
from typing import Any, Literal

import numpy as np
import pytest

from dpt.contracts import ContractError
from dpt.transport.derivatives import TransportParameter, derivative_histories
from dpt.transport.graph import prepare_transport_graph
from dpt.transport.model import IncompleteHistoryError, TransportError
from dpt.transport.rng import HistoryBatch
from dpt.transport.simulation import simulate
from helpers.transport_graph_inputs import graph_scene

wp: Any = importlib.import_module("warp")
pytestmark = pytest.mark.gpu


@pytest.mark.parametrize("estimator", ["analogue", "continuous-absorption"])
@pytest.mark.parametrize("strategy", ["plain", "grouped"])
@pytest.mark.parametrize("cone", [False, True])
def test_fixed_batch_graph_replay_matches_eager_and_changed_density(
    estimator: Literal["analogue", "continuous-absorption"],
    strategy: Literal["plain", "grouped"],
    cone: bool,
) -> None:
    graph_ws, density, source = graph_scene(
        "cuda:0", estimator=estimator, strategy=strategy, cone=cone
    )
    eager_ws, eager_density, _ = graph_scene(
        "cuda:0", estimator=estimator, strategy=strategy, cone=cone
    )
    batch = HistoryBatch(734923, 2**47, 257, "one canonical library source")
    graph = prepare_transport_graph(density, source=source, batch=batch, workspace=graph_ws)
    pointers = (graph_ws.mean.ptr, graph_ws.estimated_variance_of_mean.ptr)
    retained_graph = graph._graph
    for multiplier in (1.0, 1.0, 1.3):
        density.fill_(multiplier)
        eager_density.fill_(multiplier)
        expected = simulate(eager_density, source=source, batch=batch, workspace=eager_ws)
        actual = graph.replay()
        assert actual.estimated_variance_of_mean is not None
        assert expected.estimated_variance_of_mean is not None
        assert actual.batch == expected.batch == graph.batch
        assert actual.status_histogram == expected.status_histogram
        assert actual.status_histogram.total == batch.count
        assert (actual.mean.ptr, actual.estimated_variance_of_mean.ptr) == pointers
        assert graph._graph is retained_graph
        for name in graph_ws._histories:
            np.testing.assert_array_equal(
                graph_ws._histories[name].numpy(), eager_ws._histories[name].numpy()
            )
        np.testing.assert_array_equal(graph_ws._position.numpy(), eager_ws._position.numpy())
        np.testing.assert_allclose(
            actual.mean.numpy(), expected.mean.numpy(), rtol=3e-13, atol=1e-15
        )
        np.testing.assert_allclose(
            actual.estimated_variance_of_mean.numpy(),
            expected.estimated_variance_of_mean.numpy(),
            rtol=3e-13,
            atol=1e-18,
        )
        # Independent complete-history arithmetic: no production moment helper.
        scores, pixels = (
            graph_ws._histories["out_score"].numpy(),
            graph_ws._histories["out_pixel"].numpy(),
        )
        for pixel in range(4):
            values = [float(x) if p == pixel else 0.0 for x, p in zip(scores, pixels, strict=True)]
            mean = math.fsum(values) / batch.count
            variance = math.fsum((x - mean) ** 2 for x in values) / batch.count / (batch.count - 1)
            assert actual.mean.numpy()[pixel] == pytest.approx(mean, rel=3e-13, abs=1e-15)
            assert actual.estimated_variance_of_mean.numpy()[pixel] == pytest.approx(
                variance, rel=3e-13, abs=1e-18
            )


def test_graph_replay_checks_dynamic_inputs_sticky_status_and_binding_changes() -> None:
    workspace, density, source = graph_scene("cuda:0")
    graph = prepare_transport_graph(
        density, source=source, batch=HistoryBatch(91, 0, 257), workspace=workspace
    )
    density.fill_(0.0)
    with pytest.raises(TransportError, match="invalid device"):
        graph.replay()
    density.fill_(1.0)
    with pytest.raises(TransportError, match="invalid device"):
        graph.replay()  # A valid later input cannot erase a failed previous run.
    workspace.transport.clear_status()
    assert graph.replay().status_histogram.total == 257
    workspace.mean = wp.empty_like(workspace.mean)
    with pytest.raises(TransportError, match="bindings changed"):
        graph.replay()


def test_graph_replay_preserves_nondefault_stream_and_allocates_no_arrays(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stream = wp.Stream(device="cuda:0")
    workspace, density, source = graph_scene("cuda:0", stream=stream)
    graph = prepare_transport_graph(
        density, source=source, batch=HistoryBatch(913, 0, 257), workspace=workspace, stream=stream
    )

    def allocation(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("prepared graph replay allocated a Warp array")

    for name in ("empty", "zeros", "ones", "full", "empty_like", "zeros_like", "clone"):
        monkeypatch.setattr(wp, name, allocation)
    assert graph.replay(stream=stream).status_histogram.total == 257
    with pytest.raises(ContractError, match="owning stream"):
        graph.replay(stream=wp.Stream(device="cuda:0"))


def test_graph_preparation_delivers_existing_failed_history_status() -> None:
    workspace, density, source = graph_scene("cuda:0")
    workspace.transport._status.fill_(1 << 2)
    with pytest.raises(IncompleteHistoryError, match="budget exhausted"):
        prepare_transport_graph(
            density, source=source, batch=HistoryBatch(41, 0, 257), workspace=workspace
        )


@pytest.mark.parametrize("estimator", ["analogue", "continuous-absorption"])
def test_replayed_history_identity_preserves_supported_derivatives(
    estimator: Literal["analogue", "continuous-absorption"],
) -> None:
    workspace, density, source = graph_scene("cuda:0", estimator=estimator)
    eager_workspace, eager_density, _ = graph_scene("cuda:0", estimator=estimator)
    batch = HistoryBatch(49391, 2**51 + 317, 257)
    graph = prepare_transport_graph(
        density, source=source, batch=batch, workspace=workspace, source_amplitude=2.5
    )
    # Allocation belongs to test setup; the graph captures forward moments.
    derivatives = [wp.empty(batch.count, dtype=wp.float64, device="cuda:0") for _ in range(2)]
    cotangent = (2.0, -3.0, 1.0, -5.0)
    for multiplier in (1.0, 1.3):
        density.fill_(multiplier)
        eager_density.fill_(multiplier)
        eager = simulate(
            eager_density,
            source=source,
            batch=batch,
            workspace=eager_workspace,
            source_amplitude=2.5,
        )
        replayed = graph.replay()
        assert replayed.batch == eager.batch
        for parameter in (
            TransportParameter("log-material-density", 0),
            TransportParameter("source-amplitude"),
        ):
            contractions: list[float] = []
            for current, current_density, result, derivative in zip(
                (workspace, eager_workspace),
                (density, eager_density),
                (replayed, eager),
                derivatives,
                strict=True,
            ):
                derivative_histories(
                    current._position,
                    current._direction,
                    current._weight,
                    current_density,
                    parameter=parameter,
                    batch=result.batch,
                    workspace=current.transport,
                    out_pixel=current._histories["out_pixel"],
                    out_derivative=derivative,
                    out_status=current._histories["out_status"],
                    source_amplitude=2.5,
                )
                pixel = current._histories["out_pixel"].numpy()
                contractions.append(
                    math.fsum(
                        cotangent[int(p)] * float(d) if p >= 0 else 0.0
                        for p, d in zip(pixel, derivative.numpy(), strict=True)
                    )
                    / batch.count
                )
            np.testing.assert_array_equal(derivatives[0].numpy(), derivatives[1].numpy())
            assert contractions[0] == contractions[1]
