"""Check built DPT distributions and import the wheel in a clean environment."""

from __future__ import annotations

import argparse
import email.policy
import hashlib
import json
import re
import subprocess
import sys
import tarfile
import tempfile
import tomllib
import venv
import zipfile
from email.parser import BytesParser
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
LICENCES = (
    "LICENSE",
    "python/dpt/transport/data/JOE_KUO_LICENSE.txt",
    "python/dpt/transport/data/SCIPY_LICENSE.txt",
    "python/dpt/transport/data/XRAYLIB_LICENSE.txt",
)
FINAL_VERSION = re.compile(r"(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)")
FORBIDDEN_PARTS = {
    ".aws",
    ".codex",
    ".env",
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".ssh",
    ".venv",
    "__pycache__",
    "node_modules",
}
SENSITIVE_CONTENT = (
    re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"),
    re.compile(rb"\bgh[pousr]_[A-Za-z0-9]{36,}\b"),
    re.compile(rb"\bgithub_pat_[A-Za-z0-9_]{50,}\b"),
    re.compile(rb"\bpypi-[A-Za-z0-9_-]{40,}\b"),
    re.compile(rb"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    re.compile(rb"/(?:Users|home)/[A-Za-z0-9_.-]+/(?:Documents|\.ssh|\.config)/"),
    re.compile(rb"\b[a-zA-Z0-9.-]+\.(?:ts\.net|glddns\.com)\b"),
)
EXCLUDED_SOURCE_FILES = {
    "task_plan.md",
    "notes.md",
    "docs/about/provenance.md",
    "docs/about/licensing.md",
}
PACKAGE_RESOURCES = {
    "dpt/transport/data/sobol_joe_kuo_6_21201.npz": (
        "4859931147d42ce465b8605cb277f957d98b839d03194fdf06579357906d193b"
    ),
    "dpt/transport/data/JOE_KUO_LICENSE.txt": (
        "9d10226b50eeb34be0ab06bfa3392c7bd1f04bf602f9af4343295d1fd003d0e3"
    ),
    "dpt/transport/data/SCIPY_LICENSE.txt": (
        "221e59f5e910fd7f94e44f0dac77436a11338c285c6346232e4a850a50da0e94"
    ),
    "dpt/transport/data/BIGGS_OCCUPANCIES.json": (
        "40d12564ffd30957d8953f582d8de1cf842561f5f85786741fd721d1dbc775dc"
    ),
    "dpt/transport/data/XRAYLIB_LICENSE.txt": (
        "dc793dac39e5a3744b935b14e4658c4d880ba799d39f24a9a1424b32120ca8fb"
    ),
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def check_member(name: str, data: bytes) -> None:
    path = PurePosixPath(name)
    require(not path.is_absolute() and ".." not in path.parts, f"Unsafe archive path: {name}")
    require("\\" not in name, f"Non-POSIX archive path: {name}")
    require(not FORBIDDEN_PARTS.intersection(path.parts), f"Cache or private file: {name}")
    require(path.suffix not in {".pyc", ".pyo", ".pem", ".key"}, f"Unexpected file: {name}")
    require(len(data) <= 64 * 1024 * 1024, f"Oversized archive member: {name}")
    for pattern in SENSITIVE_CONTENT:
        require(pattern.search(data) is None, f"Sensitive content in archive member: {name}")


def read_wheel(path: Path) -> dict[str, bytes]:
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        require(len(names) == len(set(names)), "Duplicate wheel archive members")
        members = {}
        for info in archive.infolist():
            require(not info.flag_bits & 1, f"Encrypted wheel member: {info.filename}")
            if not info.is_dir():
                require(info.file_size <= 64 * 1024 * 1024, "Oversized wheel member")
                data = archive.read(info)
                check_member(info.filename, data)
                members[info.filename] = data
        return members


def read_sdist(path: Path, prefix: str) -> dict[str, bytes]:
    with tarfile.open(path, "r:gz") as archive:
        names = archive.getnames()
        require(len(names) == len(set(names)), "Duplicate source archive members")
        members = {}
        for info in archive.getmembers():
            require(info.isfile() or info.isdir(), f"Linked or special source member: {info.name}")
            require(info.name.startswith(prefix + "/"), f"Unexpected source root: {info.name}")
            name = info.name.removeprefix(prefix + "/")
            check_member(name, b"")
            if info.isfile():
                require(info.size <= 64 * 1024 * 1024, f"Oversized source member: {name}")
                stream = archive.extractfile(info)
                require(stream is not None, f"Missing source member: {name}")
                assert stream is not None
                data = stream.read()
                check_member(name, data)
                members[name] = data
        return members


def check_source_layout(members: dict[str, bytes]) -> None:
    root_files = {
        ".gitignore",
        "CITATION.bib",
        "CITATION.cff",
        "PKG-INFO",
        "README.md",
        "pyproject.toml",
        "uv.lock",
        "mkdocs.yml",
        *LICENCES,
    }
    directories = ("python/dpt/", "tests/python/", "benchmarks/", "examples/", "docs/", "tools/")
    for name in members:
        require(name not in EXCLUDED_SOURCE_FILES, f"External-only source file: {name}")
        require(
            name in root_files or name.startswith(directories), f"Unexpected source file: {name}"
        )
        if name != "PKG-INFO":
            source = ROOT / name
            require(source.is_file(), f"Source archive file absent from checkout: {name}")
            require(
                source.read_bytes() == members[name],
                f"Source archive differs from checkout: {name}",
            )
    for name in (
        "PKG-INFO",
        "README.md",
        "CITATION.cff",
        "CITATION.bib",
        "pyproject.toml",
        *LICENCES,
    ):
        require(name in members, f"Required source file missing: {name}")
    for source in (ROOT / "python/dpt").rglob("*"):
        if source.is_file() and "__pycache__" not in source.parts:
            relative = source.relative_to(ROOT).as_posix()
            require(relative in members, f"Package source missing from sdist: {relative}")


def check_metadata(data: bytes, project: dict, readme: bytes, label: str) -> None:
    metadata = BytesParser(policy=email.policy.default).parsebytes(data)
    expected = {
        "Name": project["name"],
        "Version": project["version"],
        "Requires-Python": project["requires-python"],
        "License-Expression": project["license"],
        "Description-Content-Type": "text/markdown",
    }
    for key, value in expected.items():
        require(len(metadata.get_all(key, [])) == 1, f"{label}: missing or duplicate {key}")
        require(metadata.get(key) == value, f"{label}: incorrect {key}")
    require(
        set(metadata.get_all("License-File", [])) == set(LICENCES), f"{label}: licence files differ"
    )
    description = metadata.get_payload(decode=True)
    require(isinstance(description, bytes), f"{label}: invalid README payload")
    assert isinstance(description, bytes)
    require(
        description.decode("utf-8").rstrip() == readme.decode("utf-8").rstrip(),
        f"{label}: README differs",
    )


def check_wheel_layout(members: dict[str, bytes], sdist: dict[str, bytes], stem: str) -> None:
    metadata_root = f"{stem}.dist-info/"
    metadata_files = {"METADATA", "WHEEL", "RECORD", "entry_points.txt"}
    metadata_files.update(f"licenses/{name}" for name in LICENCES)
    for name, data in members.items():
        if name.startswith("dpt/"):
            require(
                name.endswith((".py", ".md", "/py.typed")) or name in PACKAGE_RESOURCES,
                f"Unexpected package file: {name}",
            )
            source_name = "python/" + name
            require(source_name in sdist, f"Wheel file missing from sdist: {name}")
            require(data == sdist[source_name], f"Wheel differs from source archive: {name}")
        else:
            require(name.startswith(metadata_root), f"Unexpected wheel root: {name}")
            relative = name.removeprefix(metadata_root)
            require(relative in metadata_files, f"Unexpected wheel metadata file: {name}")
    for name in metadata_files:
        require(metadata_root + name in members, f"Required wheel metadata missing: {name}")
    for name in LICENCES:
        require(
            members[f"{metadata_root}licenses/{name}"] == sdist[name],
            f"Wheel licence differs: {name}",
        )
    for name, expected in PACKAGE_RESOURCES.items():
        require(name in members, f"Required attributed package resource missing: {name}")
        require(
            hashlib.sha256(members[name]).hexdigest() == expected,
            f"Attributed package resource hash differs: {name}",
        )
    for name in ("dpt/__init__.py", "dpt/py.typed", "dpt/cli.py"):
        require(name in members, f"Required package file missing: {name}")
    wheel_metadata = BytesParser(policy=email.policy.default).parsebytes(
        members[metadata_root + "WHEEL"]
    )
    require(wheel_metadata.get("Root-Is-Purelib") == "true", "Wheel must be pure Python")
    require(wheel_metadata.get_all("Tag") == ["py3-none-any"], "Unexpected wheel platform tags")


def check_installed_version(wheel: Path, version: str) -> None:
    with tempfile.TemporaryDirectory(prefix="dpt-release-import-") as temporary:
        directory = Path(temporary)
        environment = directory / "venv"
        venv.EnvBuilder(with_pip=True).create(environment)
        python = environment / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
        command = [
            str(python),
            "-I",
            "-m",
            "pip",
            "--isolated",
            "install",
            "--disable-pip-version-check",
            "--no-index",
            "--no-deps",
            str(wheel),
        ]
        subprocess.run(command, cwd=directory, check=True, capture_output=True, text=True)
        probe = (
            "import dpt, json, sys; "
            "print(json.dumps({'version': dpt.__version__, "
            "'optional_modules': sorted(set(sys.modules) & {'numpy', 'torch', 'warp'})}))"
        )
        result = subprocess.run(
            [str(python), "-I", "-c", probe],
            cwd=directory,
            check=True,
            capture_output=True,
            text=True,
        )
        identity = json.loads(result.stdout)
        require(identity["version"] == version, "Installed dpt.__version__ differs from release")
        require(not identity["optional_modules"], "Base import loads optional dependencies")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dist", type=Path, default=ROOT / "dist", help="Built distributions directory"
    )
    parser.add_argument("--tag", help="Release tag; must be v followed by the package version")
    arguments = parser.parse_args()
    try:
        project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
        version = project["version"]
        require(
            FINAL_VERSION.fullmatch(version) is not None, "Release version must have form N.N.N"
        )
        if arguments.tag is not None:
            require(arguments.tag == "v" + version, "Release tag differs from package version")
        stem = project["name"].replace("-", "_") + "-" + version
        distribution = arguments.dist.resolve()
        wheel = distribution / f"{stem}-py3-none-any.whl"
        sdist_path = distribution / f"{stem}.tar.gz"
        require(distribution.is_dir(), "Distributions directory is missing")
        require(
            set(distribution.iterdir()) == {wheel, sdist_path},
            "Expected exactly one wheel and one sdist",
        )
        wheel_members = read_wheel(wheel)
        source_members = read_sdist(sdist_path, stem)
        check_source_layout(source_members)
        check_wheel_layout(wheel_members, source_members, stem)
        for label, metadata in (
            ("wheel", wheel_members[f"{stem}.dist-info/METADATA"]),
            ("sdist", source_members["PKG-INFO"]),
        ):
            check_metadata(metadata, project, source_members["README.md"], label)
        check_installed_version(wheel, version)
        print(
            json.dumps(
                {
                    "name": project["name"],
                    "version": version,
                    "tag": arguments.tag,
                    "wheel_members": len(wheel_members),
                    "sdist_members": len(source_members),
                    "sha256": {
                        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                        for path in (wheel, sdist_path)
                    },
                    "clean_install_import": "passed",
                    "distribution_contents": "passed",
                },
                indent=2,
            )
        )
    except (
        ValueError,
        OSError,
        subprocess.CalledProcessError,
        tarfile.TarError,
        zipfile.BadZipFile,
    ) as error:
        print(f"Release check failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
