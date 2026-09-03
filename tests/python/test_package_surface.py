"""The installed package contains scientific operators and working examples."""

import importlib.util


def test_scientific_validation_and_examples_remain_importable() -> None:
    for module in (
        "dpt.validation.projection",
        "dpt.validation.transmission",
        "dpt.validation.spectral",
        "dpt.validation.transport",
        "dpt.examples.quickstart",
        "dpt.examples.registration",
        "dpt.examples.reconstruction",
        "dpt.examples.acquisition_design",
    ):
        assert importlib.util.find_spec(module) is not None
