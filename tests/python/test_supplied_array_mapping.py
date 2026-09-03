"""Array loading preserves values and hashes without whole-file copies."""

import hashlib
import importlib
import json
from pathlib import Path
from typing import Any

import pytest

from dpt.contracts import ContractError
from dpt.examples._common import ExampleInputError, load_case

np: Any = importlib.import_module("numpy")


def test_supplied_arrays_are_read_only_mapped_and_hash_checked(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "array.npy"
    values = np.arange(32, dtype=np.float32).reshape(4, 8)
    np.save(path, values, allow_pickle=False)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    config = tmp_path / "case.json"
    config.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "arrays": {
                    "values": {
                        "path": path.name,
                        "sha256": digest,
                        "units": "counts",
                        "source": "analytic regression fixture",
                        "rights": "test-owned",
                    }
                },
            }
        )
    )
    case = load_case(config)
    original = Path.read_bytes

    def no_array_copy(selected: Path) -> bytes:
        if selected == path:
            raise AssertionError("numeric inputs must be mapped, not read_bytes")
        return original(selected)

    monkeypatch.setattr(Path, "read_bytes", no_array_copy)
    loaded = case.array("values", units="counts", shape=(4, 8))
    assert isinstance(loaded, np.memmap) and not loaded.flags.writeable
    np.testing.assert_array_equal(loaded, values)
    with path.open("r+b") as output:
        output.seek(-4, 2)
        output.write(b"\0\0\0\0")
    with pytest.raises(ExampleInputError, match="sha256"):
        case.array("values", units="counts")
    assert issubclass(ExampleInputError, ContractError)
