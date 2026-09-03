"""Render package metadata and citation downloads from the root citation files."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import yaml
from mkdocs.structure.files import File


def on_config(config):
    root = Path(config.config_file_path).resolve().parent
    citation = yaml.safe_load((root / "CITATION.cff").read_text(encoding="utf-8"))
    bibtex = (root / "CITATION.bib").read_text(encoding="utf-8").strip()
    author = citation["authors"][0]
    display_name = f"{author['given-names']} {author['family-names']}"
    initials = " ".join(f"{part[0]}." for part in author["given-names"].split())
    year = re.search(r"\byear\s*=\s*[\{\"](\d{4})[\}\"]", bibtex)
    if year is None:
        raise ValueError("CITATION.bib must contain a four-digit year")
    title = citation["title"]
    version = str(citation["version"])
    url = citation["repository-code"]
    text = (
        f"{author['family-names']}, {initials} ({year[1]}). "
        f"{title} (version {version}) [Computer software]. {url}"
    )
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    if version != project["version"] or display_name != project["authors"][0]["name"]:
        raise ValueError("CITATION.cff author and version must match pyproject.toml")
    index = (root / "docs/index.md").read_text(encoding="utf-8")
    if not index.startswith(f"# {title}\n") or (
        f'{display_name} <span class="dpt-version">Version {version}</span>' not in index
    ):
        raise ValueError("The manual title, author and version must match CITATION.cff")
    for path in ("README.md", "docs/citation.md"):
        markdown = (root / path).read_text(encoding="utf-8")
        if f"```bibtex\n{bibtex}\n```" not in markdown:
            raise ValueError(f"The BibTeX block in {path} must match CITATION.bib")
    if text not in (root / "docs/citation.md").read_text(encoding="utf-8"):
        raise ValueError("The manual citation must match the root citation files")
    config.site_name = title
    config.site_author = display_name
    config.extra["dpt"] = {
        "title": title,
        "author": display_name,
        "version": version,
        "citation": text,
        "bibtex": bibtex,
    }
    return config


def on_files(files, config):
    root = Path(config.config_file_path).resolve().parent
    for name in ("CITATION.cff", "CITATION.bib"):
        files.append(File.generated(config, name, abs_src_path=str(root / name)))
    return files


def on_page_markdown(markdown, page, config, files):
    if page.file.src_uri == "citation.md":
        for name in ("CITATION.cff", "CITATION.bib"):
            markdown = markdown.replace(f"(../{name})", f"({name})")
    return markdown
