"""Download a selected reference dataset after checking its licence and hashes.

No reference data are bundled or downloaded on import. The catalogue pins an
xraylib density file and its BSD notice to one upstream commit. A cache entry
contains the original bytes, licence and a portable provenance manifest.

    >>> from dpt.reference_data import XRAYLIB_ELEMENTAL_DENSITIES, fetch_reference_data
    >>> # An explicit network operation into a caller-selected cache:
    >>> # reference = fetch_reference_data(XRAYLIB_ELEMENTAL_DENSITIES, "./reference-cache")

Source: https://github.com/tschoonj/xraylib
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from dpt.contracts import ContractError, finite_scalar, integer
from dpt.materials import Provenance

# Public boundaries also validate untyped callers at runtime.
# pyright: reportUnnecessaryIsInstance=false


def _https_url(value: str, name: str) -> None:
    if not isinstance(value, str):
        raise ContractError(f"{name} must be an HTTPS URL")
    parsed = urlsplit(value)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.fragment
    ):
        raise ContractError(f"{name} must be an HTTPS URL without embedded credentials/fragments")


@dataclass(frozen=True, slots=True)
class ReferenceDataset:
    """A reviewed source, redistribution licence and exact source-byte identity.

    ``maximum_bytes`` bounds each network read. Catalogue entries record the
    upstream commit; custom descriptors require the caller to supply verified
    rights and attribution for the particular data, rather than the host site.
    """

    name: str
    url: str
    sha256: str
    maximum_bytes: int
    licence_url: str
    licence_sha256: str
    rights: str
    citation: str
    description: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.name, str)
            or re.fullmatch(r"[a-z0-9][a-z0-9-]{0,79}", self.name) is None
        ):
            raise ContractError("dataset name must be a short lowercase identifier")
        _https_url(self.url, "reference source")
        _https_url(self.licence_url, "reference licence")
        integer(self.maximum_bytes, "maximum_bytes", minimum=1, maximum=64 * 1024 * 1024)
        for label, value in (("data", self.sha256), ("licence", self.licence_sha256)):
            if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
                raise ContractError(f"reference {label} SHA256 must contain 64 hexadecimal digits")
        for label in ("rights", "citation", "description"):
            value = getattr(self, label)
            if not isinstance(value, str) or not value.strip():
                raise ContractError(f"reference {label} must be supplied")
        if len(_manifest(self)) > 16384:
            raise ContractError("reference provenance manifest exceeds 16 KiB")

    @property
    def provenance(self) -> Provenance:
        """Original data identity and its reviewed redistribution terms."""
        return Provenance(self.url, self.sha256, self.rights, self.description)


@dataclass(frozen=True, slots=True)
class CachedReference:
    """Paths to verified source bytes, licence and manifest in a selected cache."""

    data_path: Path
    licence_path: Path
    manifest_path: Path
    dataset: ReferenceDataset


def _manifest(dataset: ReferenceDataset) -> bytes:
    return (
        json.dumps(
            {
                "name": dataset.name,
                "source": dataset.url,
                "sha256": dataset.sha256,
                "licence_source": dataset.licence_url,
                "licence_sha256": dataset.licence_sha256,
                "rights": dataset.rights,
                "citation": dataset.citation,
                "description": dataset.description,
            },
            sort_keys=True,
            indent=2,
        )
        + "\n"
    ).encode("utf-8")


_UPSTREAM = "https://raw.githubusercontent.com/tschoonj/xraylib/"
_COMMIT = "f94a3f5008dfd1c882b88ff26cd5052559423c83"
XRAYLIB_ELEMENTAL_DENSITIES = ReferenceDataset(
    name="xraylib-elemental-densities",
    url=f"{_UPSTREAM}{_COMMIT}/data/densities.dat",
    sha256="f81b55ef50de2749cc7dea47431134aa15287a53942dde9f48ab3d3bdf379e6c",
    maximum_bytes=16384,
    licence_url=f"{_UPSTREAM}{_COMMIT}/license_all.txt",
    licence_sha256="dc793dac39e5a3744b935b14e4658c4d880ba799d39f24a9a1424b32120ca8fb",
    rights="xraylib BSD three-clause licence; retain the accompanying copyright and licence notice",
    citation="Golosio, Brunetti, Sanchez del Rio, Schoonjans and Ikonen; xraylib densities.dat",
    description=f"Elemental reference density in g/cm^3 at upstream xraylib commit {_COMMIT}",
)


class _VerifiedRedirect(HTTPRedirectHandler):
    def redirect_request(
        self, req: Any, fp: Any, code: int, msg: str, headers: Any, newurl: str
    ) -> Request | None:
        _https_url(newurl, "reference redirect")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _read_url(url: str, *, maximum_bytes: int, timeout: float) -> bytes:
    request = Request(url, headers={"User-Agent": "DPT-reference-data/0.1"})
    with build_opener(_VerifiedRedirect()).open(request, timeout=timeout) as response:
        _https_url(response.geturl(), "reference redirect")
        payload = response.read(maximum_bytes + 1)
    if not payload or len(payload) > maximum_bytes:
        raise ContractError("reference response is empty or exceeds maximum_bytes")
    return payload


def _verify(payload: bytes, digest: str, label: str) -> None:
    if hashlib.sha256(payload).hexdigest() != digest:
        raise ContractError(f"reference {label} SHA256 does not match its reviewed descriptor")


def _publish_directory(staging: Path, destination: Path) -> None:
    """Atomically rename a complete directory, refusing *any* existing destination.

    Linux RENAME_NOREPLACE and Darwin RENAME_EXCL supply the same exclusion
    guarantee as Windows os.rename. Unsupported filesystems are reported rather
    than falling back to a rename that could overwrite an unrelated directory.
    """
    if os.name == "nt":
        os.rename(staging, destination)
        return
    import ctypes

    library: Any = ctypes.CDLL(None, use_errno=True)
    source, target = os.fsencode(staging), os.fsencode(destination)
    if sys.platform == "darwin":
        # https://github.com/apple-oss-distributions/xnu/blob/main/bsd/sys/stdio.h
        function = library.renamex_np
        function.argtypes = (ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint)
        function.restype = ctypes.c_int
        result = function(source, target, 4)  # RENAME_EXCL
    elif sys.platform.startswith("linux") and hasattr(library, "renameat2"):
        # https://man7.org/linux/man-pages/man2/rename.2.html
        function = library.renameat2
        function.argtypes = (
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        )
        function.restype = ctypes.c_int
        result = function(-100, source, -100, target, 1)  # AT_FDCWD, RENAME_NOREPLACE
    else:
        raise RuntimeError(
            "atomic exclusive reference-cache publication is unavailable on this platform"
        )
    if result:
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code), str(destination))


def _cached(destination: Path, dataset: ReferenceDataset) -> CachedReference:
    if destination.is_symlink() or not destination.is_dir():
        raise ContractError("reference cache entry must be an owned ordinary directory")
    data, licence, manifest = (
        destination / name for name in ("data", "LICENSE", "provenance.json")
    )
    for path, bound in ((data, dataset.maximum_bytes), (licence, 65536), (manifest, 16384)):
        if path.is_symlink() or not path.is_file() or path.stat().st_size > bound:
            raise ContractError("reference cache entry is incomplete or has unexpected files")
    _verify(data.read_bytes(), dataset.sha256, "data")
    _verify(licence.read_bytes(), dataset.licence_sha256, "licence")
    if manifest.read_bytes() != _manifest(dataset):
        raise ContractError("reference cache manifest differs from its reviewed descriptor")
    return CachedReference(data, licence, manifest, dataset)


def fetch_reference_data(
    dataset: ReferenceDataset,
    cache_directory: str | Path,
    *,
    timeout_seconds: float = 20,
) -> CachedReference:
    """SHA256(data)=dataset.sha256 and SHA256(licence)=dataset.licence_sha256.

    ``dataset`` is one reviewed ``ReferenceDataset`` descriptor identifying
    original source bytes, licence, rights and citation. ``cache_directory``
    is a caller-selected path. Return a ``CachedReference`` with paths to
    ``data``, ``LICENSE`` and ``provenance.json`` under the descriptor's
    name and hash prefix. The loader preserves byte content, including the
    physical units documented by the source, and performs no numerical
    conversion, precision change or differentiation.

    A valid cache entry is reverified and reused without network access.
    Otherwise the licence is fetched before data and both hashes must match
    before an exclusive atomic directory publication. Data reads are bounded
    by ``dataset.maximum_bytes``, licence reads by 64 KiB and each request by
    ``timeout_seconds`` in (0,300]. Invalid existing entries are reported for
    inspection and failed downloads leave no completed entry.

    """
    if not isinstance(dataset, ReferenceDataset):
        raise ContractError("dataset must be a reviewed ReferenceDataset record")
    timeout = finite_scalar(timeout_seconds, "timeout_seconds", minimum=0)
    if not 0 < timeout <= 300:
        raise ContractError("timeout_seconds must lie in (0, 300]")
    cache = Path(cache_directory)
    if cache.is_symlink():
        raise ContractError("reference cache root must not be a symlink")
    cache.mkdir(parents=True, exist_ok=True)
    destination = cache / f"{dataset.name}-{dataset.sha256[:16]}"
    if destination.exists() or destination.is_symlink():
        return _cached(destination, dataset)
    licence = _read_url(dataset.licence_url, maximum_bytes=65536, timeout=timeout)
    _verify(licence, dataset.licence_sha256, "licence")
    data = _read_url(dataset.url, maximum_bytes=dataset.maximum_bytes, timeout=timeout)
    _verify(data, dataset.sha256, "data")
    # All files become visible together; concurrent fetches cannot observe a partial entry.
    with tempfile.TemporaryDirectory(prefix=".dpt-reference-", dir=cache) as temporary:
        staging = Path(temporary)
        for name, payload in (
            ("data", data),
            ("LICENSE", licence),
            ("provenance.json", _manifest(dataset)),
        ):
            (staging / name).write_bytes(payload)
        try:
            _publish_directory(staging, destination)
        except FileExistsError:
            return _cached(destination, dataset)
    return _cached(destination, dataset)
