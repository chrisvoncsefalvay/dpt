"""Verified optional downloads never overwrite unrelated or incomplete caches."""

import hashlib
from dataclasses import replace
from pathlib import Path

import pytest

from dpt.contracts import ContractError
from dpt.reference_data import XRAYLIB_ELEMENTAL_DENSITIES, ReferenceDataset, fetch_reference_data

DATA = b"original invented density fixture\n13 2.6989\n"
LICENCE = b"Original test fixture; Apache-2.0\n"
FIXTURE = ReferenceDataset(
    "fixture",
    "https://example.org/data",
    hashlib.sha256(DATA).hexdigest(),
    1024,
    "https://example.org/licence",
    hashlib.sha256(LICENCE).hexdigest(),
    "Apache-2.0",
    "Original test fixture",
    "Not physical reference data",
)


class Response:
    def __init__(self, data: bytes, url: str) -> None:
        self.data, self.url = data, url

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def geturl(self) -> str:
        return self.url

    def read(self, count: int) -> bytes:
        return self.data[:count]


def network(
    monkeypatch: pytest.MonkeyPatch,
    *,
    data: bytes = DATA,
    licence: bytes = LICENCE,
    redirect: str | None = None,
) -> list[str]:
    calls: list[str] = []

    def open_url(request, timeout):
        assert 0 < timeout <= 300
        url = request.full_url
        calls.append(url)
        return Response(licence if url.endswith("licence") else data, redirect or url)

    class Opener:
        def open(self, request, timeout):
            return open_url(request, timeout)

    monkeypatch.setattr("dpt.reference_data.build_opener", lambda *_: Opener())
    return calls


def test_licence_is_verified_before_data_and_completed_cache_is_reused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = network(monkeypatch)
    reference = fetch_reference_data(FIXTURE, tmp_path)
    assert calls == [FIXTURE.licence_url, FIXTURE.url]
    assert reference.data_path.read_bytes() == DATA
    assert reference.licence_path.read_bytes() == LICENCE
    assert FIXTURE.sha256 in reference.manifest_path.read_text()
    assert fetch_reference_data(FIXTURE, tmp_path) == reference
    assert len(calls) == 2


def test_wrong_licence_prevents_data_request_and_leaves_no_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = network(monkeypatch, licence=b"unexpected terms")
    with pytest.raises(ContractError, match="licence SHA256"):
        fetch_reference_data(FIXTURE, tmp_path)
    assert calls == [FIXTURE.licence_url]
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("data", [b"wrong source", DATA * 100, b""])
def test_wrong_or_oversized_data_never_publishes_cache(
    data: bytes, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    network(monkeypatch, data=data)
    with pytest.raises(ContractError):
        fetch_reference_data(FIXTURE, tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_unrelated_incomplete_cache_is_not_overwritten(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = network(monkeypatch)
    entry = tmp_path / f"fixture-{FIXTURE.sha256[:16]}"
    entry.mkdir()
    sentinel = entry / "unrelated"
    sentinel.write_text("keep")
    with pytest.raises(ContractError, match="incomplete"):
        fetch_reference_data(FIXTURE, tmp_path)
    assert sentinel.read_text() == "keep"
    assert calls == []


def test_corrupt_completed_cache_is_reported_without_network_or_repair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = network(monkeypatch)
    result = fetch_reference_data(FIXTURE, tmp_path)
    result.data_path.write_bytes(b"modified")
    with pytest.raises(ContractError, match="data SHA256"):
        fetch_reference_data(FIXTURE, tmp_path)
    assert result.data_path.read_bytes() == b"modified"
    assert len(calls) == 2


@pytest.mark.parametrize(
    "url",
    [
        "http://example.org/file",
        "https://user:password@example.org/file",
        "https://example.org/file#fragment",
    ],
)
def test_credentials_and_insecure_urls_are_rejected(url: str) -> None:
    with pytest.raises(ContractError, match="HTTPS"):
        replace(FIXTURE, url=url)


def test_insecure_redirect_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    network(monkeypatch, redirect="http://example.org/downgrade")
    with pytest.raises(ContractError, match="HTTPS"):
        fetch_reference_data(FIXTURE, tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_reviewed_catalogue_pins_hashes_licence_attribution_and_upstream_commit() -> None:
    item = XRAYLIB_ELEMENTAL_DENSITIES
    assert "f94a3f5008dfd1c882b88ff26cd5052559423c83" in item.url
    assert item.sha256 == "f81b55ef50de2749cc7dea47431134aa15287a53942dde9f48ab3d3bdf379e6c"
    assert "BSD" in item.rights and "Schoonjans" in item.citation
    assert item.provenance.sha256 == item.sha256


def test_symlink_cache_is_not_followed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = network(monkeypatch)
    actual = tmp_path / "actual"
    actual.mkdir()
    link = tmp_path / "link"
    link.symlink_to(actual, target_is_directory=True)
    with pytest.raises(ContractError, match="symlink"):
        fetch_reference_data(FIXTURE, link)
    assert calls == []


def test_oversized_manifest_is_rejected_before_network() -> None:
    with pytest.raises(ContractError, match="manifest exceeds"):
        replace(FIXTURE, description="x" * 20000)


def test_redirect_handler_rejects_insecure_location_before_following() -> None:
    from urllib.request import Request

    from dpt.reference_data import _VerifiedRedirect

    with pytest.raises(ContractError, match="HTTPS"):
        _VerifiedRedirect().redirect_request(
            Request(FIXTURE.url), None, 302, "redirect", {}, "http://example.org/downgrade"
        )


def test_publication_is_atomic_and_refuses_even_an_empty_unrelated_directory(
    tmp_path: Path,
) -> None:
    from dpt.reference_data import _publish_directory

    staging, destination = tmp_path / "staging", tmp_path / "destination"
    staging.mkdir()
    for name in ("data", "LICENSE", "provenance.json"):
        (staging / name).write_text(name)
    destination.mkdir()
    with pytest.raises(FileExistsError):
        _publish_directory(staging, destination)
    assert list(destination.iterdir()) == []
    assert len(list(staging.iterdir())) == 3
    destination.rmdir()
    _publish_directory(staging, destination)
    assert not staging.exists()
    assert {path.name for path in destination.iterdir()} == {"data", "LICENSE", "provenance.json"}


def test_concurrent_fetches_publish_one_complete_identical_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from concurrent.futures import ThreadPoolExecutor

    network(monkeypatch)
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(lambda _: fetch_reference_data(FIXTURE, tmp_path), range(8)))
    assert all(result == results[0] for result in results)
    assert results[0].data_path.read_bytes() == DATA
    assert len(list(tmp_path.iterdir())) == 1
