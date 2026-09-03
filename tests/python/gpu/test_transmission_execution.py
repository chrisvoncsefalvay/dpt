"""Execution contracts on real CUDA: composition, ownership and replay."""

import importlib
import math
from typing import Any

import pytest

from dpt.transmission import (
    GradientRangeError,
    TransmissionError,
    TransmissionSpec,
    optical_depth_from_removed,
    prepare_transmission,
    transmission_vjp,
    transmit,
)

approx: Any = pytest.approx  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
wp: Any = importlib.import_module("warp")
pytestmark = pytest.mark.gpu


@wp.kernel
def scale(
    source: wp.array(dtype=wp.float32),  # pyright: ignore[reportInvalidTypeForm, reportUnknownParameterType]
    target: wp.array(dtype=wp.float32),  # pyright: ignore[reportInvalidTypeForm, reportUnknownParameterType]
):
    p = wp.tid()
    target[p] = 2.0 * source[p]


@wp.kernel
def objective(
    source: wp.array(dtype=wp.float32),  # pyright: ignore[reportInvalidTypeForm, reportUnknownParameterType]
    loss: wp.array(dtype=wp.float32),  # pyright: ignore[reportInvalidTypeForm, reportUnknownParameterType]
):
    p = wp.tid()
    wp.atomic_add(loss, 0, source[p] * source[p])


def array(values: list[float], *, grad: bool = False, retain: bool = False) -> Any:
    return wp.array(
        values, dtype=wp.float32, device="cuda:0", requires_grad=grad, retain_grad=retain
    )


@pytest.mark.parametrize("retain", [False, True])
def test_tape_composes_upstream_and_downstream_and_repeats(retain: bool) -> None:
    raw = array([0.0, 0.25, 1.0], grad=True)
    depth = array([0.0] * 3, grad=True)
    counts = array([0.0] * 3, grad=True, retain=retain)
    loss = array([0.0], grad=True)
    workspace = prepare_transmission(TransmissionSpec(beam="scalar"), max_pixels=3)
    tape = wp.Tape()
    with tape:
        wp.launch(scale, dim=3, inputs=[raw], outputs=[depth])
        transmit(depth, 3.0, out_counts=counts, workspace=workspace, tape=tape)
        wp.launch(objective, dim=3, inputs=[counts], outputs=[loss])
    expected = [-4.0 * float(c) ** 2 for c in counts.numpy()]
    for _ in range(2):
        tape.backward(loss)
        workspace.check_status()
        assert raw.grad.numpy().tolist() == approx(expected, rel=5e-7)
        if retain:
            assert counts.grad.numpy().tolist() == approx(2 * counts.numpy(), rel=2e-7)
        else:
            assert counts.grad.numpy().tolist() == [0.0] * 3
        tape.zero()
        assert raw.grad.numpy().tolist() == [0.0] * 3


def test_shared_leaf_accumulates_and_external_seeds_survive() -> None:
    depth = array([0.0, 1.0], grad=True)
    beam = array([2.0], grad=True)
    first, second = array([0.0, 0.0], grad=True), array([0.0, 0.0], grad=True)
    seed = array([3.0, -1.0])
    workspace = prepare_transmission(
        TransmissionSpec(beam="device-scalar", active_beam=True), max_pixels=2
    )
    tape = wp.Tape()
    for target in (first, second):
        transmit(depth, beam, out_counts=target, workspace=workspace, tape=tape)
    for _ in range(2):
        tape.backward(grads={first: seed, second: seed})
        workspace.check_status()
        assert depth.grad.numpy().tolist() == approx([-12.0, 4.0 * 0.36787944117], rel=3e-7)
        assert beam.grad.numpy().tolist() == approx([2 * (3.0 - 0.36787944117)], rel=3e-7)
        assert seed.numpy().tolist() == [3.0, -1.0]
        tape.zero()
        assert seed.numpy().tolist() == [3.0, -1.0]


def test_inverse_tape_and_domain() -> None:
    delta = array([0.0, 0.5, 0.875], grad=True)
    depth = array([0.0] * 3, grad=True)
    seed = array([2.0, -3.0, 4.0])
    workspace = prepare_transmission(max_pixels=3)
    tape = wp.Tape()
    optical_depth_from_removed(delta, out_L=depth, workspace=workspace, tape=tape)
    tape.backward(grads={depth: seed})
    workspace.check_status()
    assert delta.grad.numpy().tolist() == [2.0, -6.0, 32.0]
    assert seed.numpy().tolist() == [2.0, -3.0, 4.0]


def test_checked_mutation_is_rescanned_and_output_is_preserved() -> None:
    depth, output = array([1.0, 2.0]), array([7.0, 7.0])
    workspace = prepare_transmission(max_pixels=2)
    transmit(depth, out_T=output, workspace=workspace)
    output.fill_(7.0)
    depth.assign(array([1.0, float("nan")]))
    with pytest.raises(TransmissionError, match=r"L\[1\]"):
        transmit(depth, out_T=output, workspace=workspace)
    assert output.numpy().tolist() == [7.0, 7.0]
    depth.fill_(0.0)
    with pytest.raises(TransmissionError):
        workspace.check_status()
    workspace.clear_status()
    transmit(depth, out_T=output, workspace=workspace)
    assert output.numpy().tolist() == [1.0, 1.0]


def test_views_aliases_layout_capacity_and_missing_grad() -> None:
    storage = array([0.0] * 8)
    workspace = prepare_transmission(max_pixels=4)
    with pytest.raises(TransmissionError, match="overlaps"):
        transmit(storage[:4], out_T=storage[2:6], workspace=workspace)
    with pytest.raises(TransmissionError, match="overlaps"):
        transmit(storage[:4], out_T=storage[4:], out_removed=storage[4:], workspace=workspace)
    with pytest.raises(TransmissionError, match="contiguous"):
        transmit(storage[::2], out_T=storage[4:], workspace=workspace)
    with pytest.raises(TransmissionError, match="capacity"):
        transmit(storage, out_T=storage, workspace=workspace)
    with pytest.raises(TransmissionError, match="binary32"):
        transmit(
            wp.zeros(4, dtype=wp.float64, device="cuda:0"), out_T=storage[4:], workspace=workspace
        )
    with pytest.raises(TransmissionError, match="one-dimensional"):
        transmit(
            wp.zeros((2, 2), dtype=wp.float32, device="cuda:0"),
            out_T=storage[4:],
            workspace=workspace,
        )
    with pytest.raises(TransmissionError, match="binary32"):
        transmit(
            wp.zeros(4, dtype=wp.float32, device="cpu"), out_T=storage[4:], workspace=workspace
        )
    with pytest.raises(TransmissionError, match="requires_grad"):
        transmit(storage[:4], out_T=storage[4:], workspace=workspace, tape=wp.Tape())
    transmit(storage[:4], out_T=storage[4:], workspace=workspace)
    assert storage.numpy().tolist() == [0.0] * 4 + [1.0] * 4


def test_empty_inputs_and_smaller_reused_scalar_workspace() -> None:
    workspace = prepare_transmission(
        TransmissionSpec(beam="device-scalar", active_beam=True), max_pixels=65537
    )
    beam, grad = array([3.0]), array([99.0])
    for count in (65537, 257, 1, 0, 513):
        depth, seed = array([0.0] * count), array([1.0] * count)
        transmission_vjp(depth, beam, cotangent_counts=seed, out_grad_n0=grad, workspace=workspace)
        assert grad.numpy().tolist() == [float(count)]
        transmit(depth, beam, out_counts=array([0.0] * count), workspace=workspace)
    assert workspace.scratch_bytes == 4 + 8 * (257 + 2 + 1)


def test_workspace_stream_ownership_and_explicit_event_handoff() -> None:
    producer, consumer = wp.Stream("cuda:0"), wp.Stream("cuda:0")
    with wp.ScopedStream(producer):
        depth, output = array([0.0, 0.0]), array([0.0, 0.0])
        depth.fill_(1.0)
        event = wp.record_event()
    workspace = prepare_transmission(max_pixels=2, stream=consumer)
    with pytest.raises(TransmissionError, match="owning stream"):
        transmit(depth, out_T=output, workspace=workspace)
    with wp.ScopedStream(consumer):
        wp.wait_event(event)
        transmit(depth, out_T=output, workspace=workspace, validate=False)
    workspace.check_status()
    assert output.numpy().tolist() == approx([0.36787944117] * 2, rel=3e-7)


def test_graph_replay_uses_current_inputs_and_preallocated_outputs() -> None:
    depth, output, seed, gradient = (array([v] * 257) for v in (0.0, 0.0, 1.0, 0.0))
    workspace = prepare_transmission(max_pixels=257)

    def evaluate() -> None:
        transmit(depth, out_T=output, workspace=workspace, validate=False)
        transmission_vjp(
            depth, cotangent_T=seed, out_grad_L=gradient, workspace=workspace, validate=False
        )

    evaluate()  # Compile and populate all factories before graph capture.
    wp.synchronize()
    pointers = (depth.ptr, output.ptr, seed.ptr, gradient.ptr)
    with wp.ScopedCapture(device="cuda:0") as capture:
        evaluate()
    for value in (0.0, 1.0, 2.0):
        depth.fill_(value)
        wp.capture_launch(capture.graph)
        workspace.check_status()
        assert output.numpy().tolist() == approx([math.exp(-value)] * 257, rel=3e-7)
        assert gradient.numpy().tolist() == approx([-math.exp(-value)] * 257, rel=3e-7)
    assert pointers == (depth.ptr, output.ptr, seed.ptr, gradient.ptr)


def test_tape_deferred_range_error_and_explicit_reset() -> None:
    depth, output = array([0.0], grad=True), array([0.0], grad=True)
    workspace = prepare_transmission(TransmissionSpec(beam="scalar"), max_pixels=1)
    tape = wp.Tape()
    transmit(depth, 3e38, out_counts=output, workspace=workspace, tape=tape)
    tape.backward(grads={output: array([2.0])})
    with pytest.raises(GradientRangeError):
        workspace.check_status()
    tape.zero()
    workspace.clear_status()
    tape.backward(grads={output: array([0.5])})
    workspace.check_status()
    assert depth.grad.numpy().tolist() == approx([-1.5e38], rel=3e-7)


@pytest.mark.parametrize("inverse", [False, True])
def test_rejects_higher_order_and_implicit_recording(inverse: bool) -> None:
    from dpt.transmission import optical_depth_from_removed_vjp

    depth, seed, gradient = array([0.5]), array([1.0]), array([0.0])
    workspace = prepare_transmission(max_pixels=1)
    with wp.Tape(), pytest.raises(TransmissionError, match="higher-order"):
        if inverse:
            optical_depth_from_removed_vjp(
                depth, seed, out_grad_delta=gradient, workspace=workspace
            )
        else:
            transmission_vjp(depth, cotangent_T=seed, out_grad_L=gradient, workspace=workspace)
    with wp.Tape(), pytest.raises(TransmissionError, match="explicitly"):
        transmit(depth, out_T=gradient, workspace=workspace)


def test_debug_dependencies_include_fixed_beam_and_view_parent(
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(wp.config, "verify_autograd_array_access", True)
    depth = array([0.0], grad=True)
    beam_parent = array([2.0, 3.0])
    beam = beam_parent[:1]
    counts = array([0.0], grad=True)
    replacement = array([4.0], grad=True)
    workspace = prepare_transmission(TransmissionSpec(beam="device-scalar"), max_pixels=1)
    tape = wp.Tape()
    with tape:
        transmit(depth, beam, out_counts=counts, workspace=workspace, tape=tape)
        wp.copy(depth, replacement)
        assert "already been read" in capfd.readouterr().err
    assert beam_parent._is_read
    tape.reset()
    assert not beam._is_read
    assert not beam_parent._is_read
    assert not depth._is_read
    # A clean tape's backward also consumes the fixed-input dependency.
    tape = wp.Tape()
    transmit(depth, beam, out_counts=counts, workspace=workspace, tape=tape)
    tape.backward(grads={counts: array([1.0])})
    workspace.check_status()
    assert not beam_parent._is_read


def test_scalar_tape_backward_graph_replay() -> None:
    depth, counts = array([0.0] * 257, grad=True), array([0.0] * 257, grad=True)
    beam, seed = array([2.0], grad=True), array([1.0] * 257)
    workspace = prepare_transmission(
        TransmissionSpec(beam="device-scalar", active_beam=True), max_pixels=257
    )
    tape = wp.Tape()
    transmit(depth, beam, out_counts=counts, workspace=workspace, tape=tape)
    tape.backward(grads={counts: seed})
    tape.zero()
    with wp.ScopedCapture(device="cuda:0") as capture:
        transmit(depth, beam, out_counts=counts, workspace=workspace, validate=False)
        tape.zero()
        tape.backward(grads={counts: seed})
    for value in (0.0, 1.0):
        depth.fill_(value)
        wp.capture_launch(capture.graph)
        workspace.check_status()
        assert depth.grad.numpy().tolist() == approx([-2 * math.exp(-value)] * 257, rel=3e-7)
        assert beam.grad.numpy().tolist() == approx([257 * math.exp(-value)], rel=3e-7)
        assert seed.numpy().tolist() == [1.0] * 257


def test_tape_backward_on_nondefault_stream() -> None:
    stream = wp.Stream("cuda:0")
    with wp.ScopedStream(stream):
        depth, output, seed = (
            array([0.0, 1.0], grad=True),
            array([0.0, 0.0], grad=True),
            array([2.0, 3.0]),
        )
        workspace = prepare_transmission(max_pixels=2, stream=stream)
        tape = wp.Tape()
        transmit(depth, out_log_T=output, workspace=workspace, tape=tape)
        tape.backward(grads={output: seed})
        completed = wp.record_event()
    wp.wait_event(completed)
    workspace.check_status()
    assert depth.grad.numpy().tolist() == [-2.0, -3.0]
    assert seed.numpy().tolist() == [2.0, 3.0]


def test_tape_accumulates_before_final_rounding() -> None:
    maximum = float.fromhex("0x1.fffffep+127")
    depth, counts = array([0.0], grad=True), array([0.0], grad=True)
    workspace = prepare_transmission(TransmissionSpec(beam="scalar"), max_pixels=1)
    tape = wp.Tape()
    transmit(depth, maximum, out_counts=counts, workspace=workspace, tape=tape)
    depth.grad.fill_(maximum)
    tape.backward(grads={counts: array([2.0])})
    workspace.check_status()
    assert depth.grad.numpy().tolist() == [-maximum]
