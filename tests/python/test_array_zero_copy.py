"""Zero-copy CPU array protocol contracts without any implicit conversion."""

import importlib
from typing import Any, ClassVar, cast

import pytest

from dpt._runtime import prepare_context
from dpt.contracts import ContractError
from dpt.interop import as_warp_array

np: Any = importlib.import_module("numpy")
wp: Any = importlib.import_module("warp")


def test_numpy_nd_array_is_aliased_and_flattened_without_allocating(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = prepare_context(device="cpu")
    values = np.arange(24, dtype=np.float32).reshape(2, 3, 4)

    def forbidden(*_: Any, **__: Any) -> Any:
        raise AssertionError("a zero-copy adapter allocated a Warp buffer")

    monkeypatch.setattr(wp, "empty", forbidden)
    monkeypatch.setattr(wp, "zeros", forbidden)
    array = as_warp_array(values, context=context, dtype=wp.float32)
    assert array.ptr == values.__array_interface__["data"][0]
    assert tuple(array.shape) == (24,)
    values.flat[4] = 91.0
    assert array.numpy()[4] == 91.0
    ranked = as_warp_array(values, context=context, dtype=wp.float32, flatten=False)
    assert tuple(ranked.shape) == (2, 3, 4)
    assert ranked.ptr == array.ptr


def test_numpy_wrong_dtype_strides_and_readonly_storage_are_rejected() -> None:
    context = prepare_context(device="cpu")
    values = np.arange(12, dtype=np.float32)
    with pytest.raises(ContractError, match="dtype"):
        as_warp_array(values.astype(np.float64), context=context, dtype=wp.float32)
    with pytest.raises(ContractError, match="contiguous"):
        as_warp_array(values[::2], context=context, dtype=wp.float32)
    values.flags.writeable = False
    with pytest.raises(ContractError, match="zero-copy Warp array view"):
        as_warp_array(values, context=context, dtype=wp.float32)


def test_existing_warp_view_preserves_storage_and_rejects_ordinary_sequences() -> None:
    context = prepare_context(device="cpu")
    array = wp.ones((2, 3), dtype=wp.float32, device="cpu")
    flat = as_warp_array(array, context=context, dtype=wp.float32)
    assert flat.ptr == array.ptr
    assert tuple(flat.shape) == (6,)
    with pytest.raises(ContractError, match="array protocol"):
        as_warp_array(cast(Any, [1.0, 2.0]), context=context, dtype=wp.float32)


def test_legacy_cuda_interface_requires_declared_device_and_never_uploads() -> None:
    context = prepare_context(device="cpu")

    class LegacyCudaArray:
        __cuda_array_interface__: ClassVar[dict[str, Any]] = {
            "shape": (2,),
            "typestr": "<f4",
            "data": (123, False),
        }

    with pytest.raises(ContractError, match="explicit source_device"):
        as_warp_array(cast(Any, LegacyCudaArray()), context=context, dtype=wp.float32)
