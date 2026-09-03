"""Scientific resources and their required notices survive packaging."""

import tomllib
from pathlib import Path

ROOT = Path(__file__).parents[2]


def test_bundled_scientific_data_include_required_licences():
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    assert metadata["name"] == "differentiable-photon-transport"
    notices = set((ROOT / "python/dpt/transport/data").glob("*LICENSE.txt"))
    assert notices
    assert {str(path.relative_to(ROOT)) for path in notices} <= set(metadata["license-files"])
    for path in notices:
        assert path.stat().st_size > 0
