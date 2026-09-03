"""Write checked NumPy inputs and a versioned, hashed application case."""

from __future__ import annotations

# The writer and reader share internal manifest validators.
# pyright: reportPrivateUsage=false
import importlib
import json
import os
import re
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from dpt.examples._common import ExampleInputError, _digest, _text


@dataclass(frozen=True, slots=True)
class CaseArray:
    """One native numeric array and its units, source and usage rights."""

    values: Any
    units: str
    source: str
    rights: str


def write_case(
    directory: str | Path,
    configuration: Mapping[str, object],
    arrays: Mapping[str, CaseArray],
) -> Path:
    """Create a new directory containing ``case.json`` and hashed .npy inputs.

    Array values retain their dtype and shape. The writer requires finite,
    native-endian, C-contiguous real arrays. It never overwrites an existing
    directory, and removes its own incomplete directory if writing fails.
    """
    np: Any = importlib.import_module("numpy")
    if not isinstance(cast(object, configuration), Mapping) or not all(
        isinstance(cast(object, key), str) for key in configuration
    ):
        raise ExampleInputError("configuration needs named JSON fields")
    if "arrays" in configuration or "schema_version" in configuration:
        raise ExampleInputError("the writer supplies arrays and schema_version")
    try:
        encoded = json.dumps(dict(configuration), allow_nan=False)
        prepared_configuration: dict[str, Any] = json.loads(encoded)
    except (TypeError, ValueError) as error:
        raise ExampleInputError("configuration must contain finite JSON values") from error
    if not isinstance(cast(object, arrays), Mapping) or not arrays:
        raise ExampleInputError("supply at least one CaseArray")
    prepared: dict[str, tuple[Any, dict[str, str]]] = {}
    for name, record in arrays.items():
        if (
            not isinstance(cast(object, name), str)
            or re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", name) is None
        ):
            raise ExampleInputError(
                "array names must start with a letter and contain letters, digits, _ or -"
            )
        if not isinstance(cast(object, record), CaseArray):
            raise ExampleInputError(f"arrays.{name} needs a CaseArray")
        data = np.asarray(record.values)
        if data.dtype.kind not in "fiu" or not data.dtype.isnative:
            raise ExampleInputError(f"arrays.{name} needs a native-endian real numeric dtype")
        if not data.flags.c_contiguous:
            raise ExampleInputError(f"arrays.{name} must be C-contiguous; convert explicitly first")
        flattened = data.reshape(-1)
        for start in range(0, int(data.size), 1024 * 1024):
            if not bool(np.isfinite(flattened[start : start + 1024 * 1024]).all()):
                raise ExampleInputError(f"arrays.{name} contains a nonfinite value")
        prepared[name] = (
            data,
            {
                "units": _text(record.units, f"arrays.{name}.units"),
                "source": _text(record.source, f"arrays.{name}.source"),
                "rights": _text(record.rights, f"arrays.{name}.rights"),
            },
        )
    destination = Path(directory).expanduser()
    try:
        destination.mkdir(mode=0o700, parents=False, exist_ok=False)
    except FileExistsError as error:
        raise ExampleInputError("case directory already exists; choose a new directory") from error
    try:
        manifest: dict[str, dict[str, str]] = {}
        for name, (data, provenance) in prepared.items():
            path = destination / f"{name}.npy"
            with path.open("xb") as handle:
                np.save(handle, data, allow_pickle=False)
                handle.flush()
                os.fsync(handle.fileno())
            manifest[name] = {"path": path.name, "sha256": _digest(path), **provenance}
        payload = {
            "schema_version": 1,
            **prepared_configuration,
            "arrays": manifest,
        }
        temporary = destination / ".case.json.tmp"
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        path = destination / "case.json"
        temporary.replace(path)
        return path.resolve()
    except BaseException:
        shutil.rmtree(destination)
        raise
