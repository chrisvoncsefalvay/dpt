"""Reject backend configuration failures before comparing numerical outputs."""

import importlib
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture(scope="module")
def reference_backend():
    path = Path(__file__).resolve().parents[2] / "benchmarks" / "reference.py"
    spec = importlib.util.spec_from_file_location("dpt_reference_backend_contracts", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class LeapConfiguration:
    """A configuration boundary with explicit rejection and retained-state checks."""

    def __init__(self, failure=None):
        self.failure = failure
        self.order = 0
        self.projector = "VD"
        self.gpu = -1
        self.libprojectors = SimpleNamespace(
            project=SimpleNamespace(), backproject=SimpleNamespace()
        )

    def version(self):
        return "configuration-fixture"

    def set_gpu(self, gpu):
        self.gpu = gpu

    def get_gpu(self):
        return self.gpu

    def set_modularbeam(self, *args):
        self.modular = args
        return True

    def set_volumeDimensionOrder(self, order):  # noqa: N802 - LEAP's public API spelling.
        self.order = order
        return self.failure != "order_rejected"

    def get_volumeDimensionOrder(self):  # noqa: N802 - LEAP's public API spelling.
        return 0 if self.failure == "order_wrong" else self.order

    def set_volume(self, *args):
        self.volume = args
        return True

    def set_projector(self, projector):
        self.projector = projector
        return self.failure != "sf_rejected"

    def get_projector(self):
        return "VD" if self.failure == "sf_wrong" else self.projector

    def set_diameterFOV(self, value):  # noqa: N802 - LEAP's public API spelling.
        self.diameter = value
        return True


def configure_imports(monkeypatch, reference, model):
    np = importlib.import_module("numpy")
    original = reference.importlib.import_module

    def load(name):
        if name == "leapctype":
            return SimpleNamespace(tomographicModels=lambda: model)
        return original(name)

    monkeypatch.setattr(reference.importlib, "import_module", load)
    return np


@pytest.mark.parametrize("failure", ["order_rejected", "order_wrong", "sf_rejected", "sf_wrong"])
def test_leap_rejects_unselected_storage_or_backprojector(monkeypatch, reference_backend, failure):
    model = LeapConfiguration(failure)
    configure_imports(monkeypatch, reference_backend, model)
    with pytest.raises(RuntimeError, match=r"declared (ZYX volume order|SF backprojector)"):
        reference_backend.LeapOperator(reference_backend.Case(16, 5, 7, 1), "cpu")


def test_leap_configuration_preserves_physical_pixel_centres(monkeypatch, reference_backend):
    model = LeapConfiguration()
    np = configure_imports(monkeypatch, reference_backend, model)
    case = reference_backend.Case(16, 5, 7, 1)
    reference_backend.LeapOperator(case, "cpu")
    assert model.order == 1
    assert model.projector == "SF"
    assert model.volume == (16, 16, 16, 1.0, 1.0, 0, 0, 0)
    assert model.diameter == 32
    views, rows, columns, row_pitch, column_pitch, sources, centres, row_axes, column_axes = (
        model.modular
    )
    assert (views, rows, columns) == (5, 7, 7)
    for index, panel in enumerate(case.panels):
        assert sources[index] == pytest.approx(panel.source_mm, abs=4e-6)
        for row, column in ((0, 0), (3, 4), (6, 6)):
            actual = (
                centres[index]
                + (row - (rows - 1) / 2) * row_pitch * row_axes[index]
                + (column - (columns - 1) / 2) * column_pitch * column_axes[index]
            )
            assert np.isfinite(actual).all()
            assert actual == pytest.approx(panel.pixel_centre(row, column), abs=4e-6)
