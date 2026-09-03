"""Production observation preparation and checkpoint restoration retain their ownership."""

from dataclasses import replace
from typing import Any, Literal

import numpy as np
import pytest

from dpt._runtime import cpu_test_context
from dpt.contracts import ContractError
from dpt.material_reconstruction import MaterialReconstruction
from helpers import material_inputs as materials
from helpers import material_solver_inputs as inputs


def prepared_inputs(
    precision: Literal["float32", "float64"] = "float64", **settings: Any
) -> dict[str, Any]:
    supplied = materials._inputs(precision)
    supplied["settings"] = replace(supplied["settings"], **settings)
    return supplied


@pytest.mark.parametrize("precision", ["float32", "float64"])
@pytest.mark.parametrize("observation", ["poisson", "signal_shared", "signal_per_pixel"])
def test_prepared_views_share_fixed_uploads_within_an_instance_only(precision, observation):
    with cpu_test_context():
        if observation == "poisson":
            supplied = prepared_inputs(precision, iterations=1)
        else:
            supplied = inputs._signal_inputs(
                heterogeneous=True, shared=observation == "signal_shared", precision=precision
            )
            supplied["device"] = "cpu"
            supplied["settings"] = replace(
                supplied["settings"],
                iterations=1,
                gradient_mapping_tolerance=0,
                relative_gradient_mapping_tolerance=0,
            )
        solver = MaterialReconstruction(**supplied)
        separate = MaterialReconstruction(**supplied)
        separate_fields = separate.fractions_numpy().copy()
        assert solver.fractions.ptr != separate.fractions.ptr
        assert (solver._path_seed is solver._path_total) == (precision == "float64")
        assert len(solver.views) == len(supplied["views"]) == 2
        for index, (resident, source) in enumerate(
            zip(solver.views, supplied["views"], strict=True)
        ):
            observations = source.counts if observation == "poisson" else source.observations
            assert resident.shape == observations.shape
            assert resident.path_seed.ptr == solver._path_seed.ptr
            assert resident.path_total.ptr == solver._path_total.ptr
            assert resident.channel_seed.ptr == solver._channel_seed.ptr
            for channel, row in enumerate(resident.channels):
                np.testing.assert_array_equal(row.counts.numpy(), observations[channel].reshape(-1))
                assert row.weights.ptr == solver.views[1 - index].channels[channel].weights.ptr
                assert row.response.ptr == solver.views[1 - index].channels[channel].response.ptr
                assert row.weights.ptr != separate.views[index].channels[channel].weights.ptr
                assert row.response.ptr != separate.views[index].channels[channel].response.ptr
                assert row.counts.ptr != solver.views[1 - index].channels[channel].counts.ptr
                if observation != "poisson":
                    np.testing.assert_array_equal(
                        row.valid.numpy(), source.valid[channel].reshape(-1)
                    )
                    np.testing.assert_array_equal(
                        row.objective_weights.numpy(), source.objective_weights[channel].reshape(-1)
                    )
        objective = solver.evaluate(gradient=True)
        assert np.isfinite(objective)
        assert np.isfinite(solver.gradient.numpy()).all()
        if observation == "poisson":
            np.testing.assert_allclose(
                objective,
                materials._independent_objective(supplied, supplied["initial_fractions"]),
                rtol=2e-5,
                atol=1e-6,
            )
        report = solver.solve(objective_id="prepared-v1")
        assert report["accepted_steps"] == 1
        assert np.isfinite(report["final_objective"])
        assert not np.array_equal(solver.fractions_numpy(), supplied["initial_fractions"])
        np.testing.assert_array_equal(separate.fractions_numpy(), separate_fields)
        np.testing.assert_array_equal(solver.checkpoint().fields, solver.fractions_numpy())


@pytest.mark.parametrize(
    ("refusal", "message"),
    [
        ("negative_count", "integers"),
        ("fractional_count", "integers"),
        ("zero_beam", "positive open-beam"),
        ("unmasked_nonfinite", "unmasked observations"),
        ("negative_weight", "nonnegative"),
        ("empty_valid", "at least one valid"),
    ],
)
def test_observation_contracts_refuse_before_device_resolution_and_allow_valid_retry(
    refusal, message
):
    signal = refusal in {"unmasked_nonfinite", "negative_weight", "empty_valid"}
    supplied = inputs._signal_inputs(shared=False) if signal else inputs._inputs()
    original = supplied["views"][0]
    if refusal == "negative_count":
        invalid = replace(original, counts=np.full_like(original.counts, -1))
    elif refusal == "fractional_count":
        invalid = replace(original, counts=original.counts + np.float32(0.5))
    elif refusal == "zero_beam":
        invalid = replace(original, weights=np.zeros_like(original.weights))
    elif refusal == "unmasked_nonfinite":
        invalid = replace(original, valid=np.ones_like(original.valid))
    elif refusal == "negative_weight":
        invalid = replace(original, objective_weights=-np.ones_like(original.objective_weights))
    else:
        invalid = replace(original, valid=np.zeros_like(original.valid))
    # An actual invalid device makes the host/device preparation order observable.
    with pytest.raises(ContractError, match=message):
        MaterialReconstruction(
            **{**supplied, "views": (invalid,), "device": "invalid-phase-device"}
        )
    with cpu_test_context():
        supplied["settings"] = replace(supplied["settings"], iterations=1)
        solver = MaterialReconstruction(**{**supplied, "device": "cpu"})
        assert np.isfinite(solver.solve()["final_objective"])


@pytest.mark.parametrize("refusal", ["identity", "counter", "history"])
def test_checkpoint_refusals_before_upload_preserve_accepted_fields_and_snapshot(refusal):
    with cpu_test_context():
        first = MaterialReconstruction(**prepared_inputs(iterations=1))
        first.solve(objective_id="prepared-v1")
        checkpoint = first.checkpoint()
        solver = MaterialReconstruction(**prepared_inputs(iterations=2))
        solver.solve(objective_id="prepared-v1")
        previous = solver.checkpoint()
        fields = solver.fractions_numpy().copy()
        if refusal == "counter":
            checkpoint = replace(checkpoint, state={**checkpoint.state, "accepted": -1})
        elif refusal == "history":
            checkpoint = replace(checkpoint, state={**checkpoint.state, "history": ()})
        with pytest.raises(ContractError):
            solver.solve(
                checkpoint=checkpoint,
                objective_id="different-v1" if refusal == "identity" else "prepared-v1",
            )
        current = solver.checkpoint()
        assert current.state == previous.state
        assert current.objective_id == previous.objective_id
        np.testing.assert_array_equal(current.fields, previous.fields)
        np.testing.assert_array_equal(solver.fractions_numpy(), fields)


def test_late_missing_voxel_refusal_uploads_fields_clears_snapshot_and_allows_fresh_retry():
    with cpu_test_context():
        supplied = prepared_inputs(iterations=2)
        first = MaterialReconstruction(**prepared_inputs(iterations=1))
        first.solve(objective_id="prepared-v1")
        checkpoint = first.checkpoint()
        scale = np.ones(supplied["grid"].shape, dtype=np.float32)
        solver = MaterialReconstruction(**{**supplied, "voxel_metric_scale": scale})
        solver.solve(objective_id="prepared-v1")
        previous = solver.checkpoint()
        assert not np.array_equal(previous.fields, checkpoint.fields)
        with pytest.raises(ContractError, match="no voxel scale"):
            solver.solve(checkpoint=checkpoint, objective_id="prepared-v1")
        np.testing.assert_array_equal(solver.fractions_numpy(), checkpoint.fields)
        with pytest.raises(ContractError, match="accepted state"):
            solver.checkpoint()
        fresh = MaterialReconstruction(
            **{
                **supplied,
                "initial_fractions": checkpoint.fields.copy(),
                "voxel_metric_scale": scale,
            }
        )
        expected = fresh.solve(objective_id="prepared-v1")
        actual = solver.solve(objective_id="prepared-v1")
        np.testing.assert_array_equal(solver.fractions_numpy(), fresh.fractions_numpy())
        assert actual["final_objective"] == expected["final_objective"]
        assert actual["history"][-1]["objective"] == expected["history"][-1]["objective"]
        assert solver.checkpoint().state["accepted_loss"] == actual["final_objective"]


@pytest.mark.parametrize("algorithm", ["bb", "inertial", "gll"])
def test_voxel_checkpoint_restore_and_accepted_callback_failure_preserve_resume(algorithm):
    with cpu_test_context():
        options: dict[str, Any] = {
            "bb": {"step_selection": "bb"},
            "inertial": {"acceleration": "inertial"},
            "gll": {"nonmonotone_window": 3},
        }[algorithm]
        supplied = prepared_inputs(iterations=4, **options)
        scale = np.linspace(0.7, 1.3, supplied["grid"].voxels, dtype=np.float32).reshape(
            supplied["grid"].shape
        )
        complete = MaterialReconstruction(**{**supplied, "voxel_metric_scale": scale})
        expected = complete.solve(objective_id="prepared-v1")
        first = MaterialReconstruction(
            **{
                **supplied,
                "settings": replace(supplied["settings"], iterations=1),
                "voxel_metric_scale": scale,
            }
        )
        first.solve(objective_id="prepared-v1")
        checkpoint = first.checkpoint()
        resumed = MaterialReconstruction(**supplied)
        failure = ValueError("accepted callback failed")
        events = []

        def fail_after_acceptance(iteration, record):
            assert iteration == 2
            assert record["accepted"]
            assert resumed.checkpoint().state["accepted"] == 2
            raise failure

        with pytest.raises(ValueError) as raised:
            resumed.solve(
                checkpoint=checkpoint,
                objective_id="prepared-v1",
                callback=fail_after_acceptance,
                observe_lifecycle=events.append,
            )
        assert raised.value is failure
        assert events[0].status == "resumed"
        assert events[-1].status == "terminal"
        accepted = resumed.checkpoint()
        assert accepted.state["accepted"] == 2
        np.testing.assert_array_equal(accepted.voxel_scale, checkpoint.voxel_scale)
        final = MaterialReconstruction(**supplied)
        actual = final.solve(checkpoint=accepted, objective_id="prepared-v1")
        assert actual["accepted_steps"] == expected["accepted_steps"] == 4
        np.testing.assert_array_equal(final.fractions_numpy(), complete.fractions_numpy())
        np.testing.assert_array_equal(final.gradient.numpy(), complete.gradient.numpy())
        assert [row["objective"] for row in actual["history"]] == [
            row["objective"] for row in expected["history"]
        ]
        for observed, original in zip(
            final.predictions_numpy(), complete.predictions_numpy(), strict=True
        ):
            np.testing.assert_array_equal(observed, original)
