"""Check manual snippets; CUDA runs and CPU kernel adaptations are explicit."""

from __future__ import annotations

import argparse
import ast
import os
import re
import sys
from contextlib import nullcontext
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HOST_PAGES = (
    "getting-started/first-calculation.md",
    "getting-started/quickstart.md",
    "guides/phantoms.md",
    "guides/beam-hardening.md",
    "guides/photon-counting.md",
    "guides/physical-transport.md",
    "guides/transport-readout.md",
    "guides/transport-designs.md",
    "guides/transport-geometry.md",
    "guides/spectral-detector.md",
    "guides/recovery.md",
    "guides/imaging-volumes.md",
)
CUDA_PAGES = (
    "guides/primary-projection.md",
    "guides/transport.md",
    "guides/prepared-projection.md",
    "guides/batched-geometry.md",
    "guides/spectral-pipeline.md",
    "guides/simulation.md",
    "guides/operator-diagnostics.md",
    "guides/detector-physics.md",
    "guides/regularization.md",
)
PYTHON_BLOCK = re.compile(r"^```python\s*\n(.*?)^```\s*$", re.MULTILINE | re.DOTALL)


def blocks(path: Path) -> list[str]:
    return PYTHON_BLOCK.findall(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument(
        "--warp-cpu", action="store_true", help="Adapt CUDA guide snippets to the CPU test context"
    )
    modes.add_argument("--cuda", action="store_true", help="Run guide snippets on actual cuda:0")
    options = parser.parse_args()
    sys.path.insert(0, str(ROOT / "python"))
    import warp as wp

    wp.config.kernel_cache_dir = os.environ.get("WARP_CACHE_PATH", str(ROOT / ".warp-cache"))

    snippets = 0
    for path in sorted((ROOT / "docs").rglob("*.md")):
        for index, code in enumerate(blocks(path), 1):
            ast.parse(code, filename=f"{path}:{index}")
            snippets += 1
    print(f"Syntax: {snippets} Python blocks passed")

    pages = HOST_PAGES + CUDA_PAGES if options.warp_cpu or options.cuda else HOST_PAGES
    for name in pages:
        path = ROOT / "docs" / name
        scope: dict[str, object] = {"__name__": "__documentation_example__"}
        context = nullcontext()
        if name in CUDA_PAGES and options.warp_cpu:
            from dpt._runtime import cpu_test_context

            context = cpu_test_context()
        with context:
            for index, code in enumerate(blocks(path), 1):
                if name in CUDA_PAGES and options.warp_cpu:
                    code = code.replace('"cuda:0"', '"cpu"')
                exec(compile(code, f"{path}:{index}", "exec"), scope)
        print(f"Passed: {name}")

    if options.warp_cpu:
        print("Guide execution: CPU kernel-test context")
    elif options.cuda:
        print("Guide execution: cuda:0")
    else:
        print("Host examples passed; CUDA snippets passed syntax checks")


if __name__ == "__main__":
    main()
