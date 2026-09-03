"""Run the clean CT child inside NVIDIA's probe-owned cache directories.

The official probe runs this command twice. Each process writes a distinct
result, retaining its same input and launches while the probe controls caches.
All timings from this entry point are instrumentation diagnostics only.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


class Tee:
    def __init__(self, original, log):
        self.original = original
        self.log = log

    def write(self, text):
        self.log.write(text)
        return self.original.write(text)

    def flush(self):
        self.log.flush()
        self.original.flush()

    def isatty(self):
        return False

    def fileno(self):
        return self.original.fileno()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    for key in ("WARP_CACHE_PATH", "WARP_CACHE_ROOT", "CUDA_CACHE_PATH"):
        if not os.environ.get(key):
            raise RuntimeError("Official NVIDIA probe must supply " + key)
    state = "instrumented_cold" if os.environ.get("_WARP_PROBE_RECORD") else "warm_process"
    name = f"{config['precision']}-{state}-{os.getpid()}-{time.time_ns()}"
    args.output.mkdir(parents=True, exist_ok=True)
    result_path = args.output / (name + ".json")
    config["result"] = str(result_path.resolve())
    config["cache_state"] = state
    config["timing_use"] = "official NVIDIA compilation diagnostic only"
    config_path = args.output / (name + "-config.json")
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    from benchmarks.cold_start import child

    with (args.output / (name + ".log")).open("x") as log:
        original_out, original_err = sys.stdout, sys.stderr
        sys.stdout = Tee(original_out, log)
        sys.stderr = Tee(original_err, log)
        try:
            child(config_path)
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            sys.stdout, sys.stderr = original_out, original_err


if __name__ == "__main__":
    main()
