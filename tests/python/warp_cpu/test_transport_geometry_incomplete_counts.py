"""Incomplete physical observations overwrite caller-owned diagnostic counts."""

# Runtime Warp arrays are intentional device records.
# pyright: reportUnknownMemberType=false, reportPrivateUsage=false
import dataclasses
import importlib
import math
from typing import Any

import numpy as np
import pytest

from dpt.geometry import DetectorGeometry, RigidTransform
from dpt.transport._constants import BUDGET_MASK, INPUT_MASK
from dpt.transport.geometry_derivatives import prepare_transport_geometry, transport_geometry_vjp
from dpt.transport.geometry_histories import GeometryPhysicalMedium
from dpt.transport.geometry_parameters import GeometryParameter
from dpt.transport.geometry_sampling import GeometrySamplingDesign
from dpt.transport.model import IncompleteHistoryError, OrientedDetector
from dpt.transport.rng import HistoryBatch
from dpt.transport.source import PointConeSource
from helpers.transport_geometry_cases import _outputs, _spec
from helpers.transport_geometry_cases import actual_aluminium as actual_aluminium

wp: Any = importlib.import_module("warp")


def test_incomplete_physical_observation_overwrites_poisoned_counts(actual_aluminium, warp_device):
    """The original seed71 ledger's failed history34 cannot retain old counts."""
    law, packed, trial_guard = "isotropic-elastic", False, 64
    energies, absorption, scattering, physical, shells = actual_aluminium
    if law == "bound-electron-compton":
        physical = dataclasses.replace(physical, bound_compton=(shells,))
    angle = 0.73
    cosine, sine = math.cos(angle), math.sin(angle)
    pose = RigidTransform((cosine, 0.0, sine, 0.0, 1.0, 0.0, -sine, 0.0, cosine))
    emitter = pose.point((0.5, -1.0, 0.0))
    target = pose.point((0.5, 3.0, 0.0))
    u, v = pose.direction((1.0, 0.0, 0.0)), pose.direction((0.0, 0.0, 1.0))
    source = PointConeSource(DetectorGeometry(emitter, target, u, v, (0.8, 2.0), (1, 1)))
    spec = dataclasses.replace(
        _spec(),
        detector=OrientedDetector(DetectorGeometry(emitter, target, u, v, (6.0, 6.0), (1, 1))),
        object_pose=pose,
        energy_kev=10.0,
        coefficient_energies_kev=energies,
        scoring="energy-kev",
        scattering_law=law,
        max_angle_trials=trial_guard,
        compton_sampler="packed-uniform-rejection" if packed else "legacy-rejection",
    )
    medium = GeometryPhysicalMedium((0,), absorption, scattering, (1.0,), (1.0,), physical)
    parameter = GeometryParameter("launch-target", "translation", 0)
    layout = (
        "bound-compton-v3"
        if law == "bound-electron-compton"
        else "packed-compton-v2"
        if packed
        else "legacy-v1"
    )
    workspace = prepare_transport_geometry(
        spec,
        source=source,
        medium=medium,
        parameters=(parameter,),
        batch=HistoryBatch(71, 0, 64),
        design=GeometrySamplingDesign("iid", 4, trial_guard, 4, layout=layout),
        device=warp_device,
    )
    outputs = _outputs(workspace)
    with workspace.context.scope():
        cotangent = wp.array([-2.0], dtype=wp.float64, device=workspace.context.device)
    # These are real output buffers consumed by the production geometry kernel.
    # Prove the nonzero initial state on each device before the failing run.
    with workspace.context.scope():
        outputs["out_counts"].fill_(wp.vec2i(17, 29))
    np.testing.assert_array_equal(
        outputs["out_counts"].numpy(),
        np.broadcast_to(np.array([17, 29], dtype=np.int32), (64, 1, 2)),
    )
    with pytest.raises(IncompleteHistoryError, match="budget exhausted"):
        transport_geometry_vjp(cotangent, workspace=workspace, **outputs)
    status = int(workspace._reference._status.numpy()[0])
    assert status & BUDGET_MASK and not status & INPUT_MASK
    bad = np.flatnonzero(~np.isfinite(outputs["out_values"].numpy()[:, 0]))
    np.testing.assert_array_equal(bad, [34])
    assert np.isnan(outputs["out_derivatives"].numpy()[bad, 0]).all()
    np.testing.assert_array_equal(outputs["out_counts"].numpy()[bad, 0], [[0, 0]])
    for name in ("out_mean", "out_gradient", "out_second_moment"):
        assert np.isnan(outputs[name].numpy()).all()
