"""Shared completed-operation timers, hardware samples and runtime metadata."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any


def utc():
    return datetime.now(UTC).isoformat()


def command(*args):
    try:
        result = subprocess.run(args, capture_output=True, text=True, check=False)
    except FileNotFoundError as error:
        return {"command": list(args), "returncode": 127, "stdout": "", "stderr": str(error)}
    return {
        "command": list(args),
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def digest_sources(source):
    entries = {
        str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted((source / "python/dpt").rglob("*.py"))
    }
    combined = hashlib.sha256(json.dumps(entries, sort_keys=True).encode()).hexdigest()
    return {"sha256": combined, "files": entries}


def load_runtime():
    """Set source information before importing DPT's Warp kernel modules."""
    import numpy as np
    import torch
    import warp as wp

    wp.config.lineinfo = True
    from dpt.imaging_io import read_nifti
    from dpt.projection import (
        ProjectionSpec,
        prepare_projection,
        project_optical_depth,
        projection_vjp,
    )

    return SimpleNamespace(
        np=np,
        torch=torch,
        wp=wp,
        read_nifti=read_nifti,
        ProjectionSpec=ProjectionSpec,
        prepare_projection=prepare_projection,
        project_optical_depth=project_optical_depth,
        projection_vjp=projection_vjp,
        utc=utc,
        command=command,
        digest_sources=digest_sources,
        platform=platform,
        importlib=importlib,
        __file__=__file__,
    )


class NvmlMonitor:
    """Sample device-wide and per-process memory, including Warp allocations.

    Peaks are explicitly sampled lower bounds, not allocator-exact maxima.
    Unsupported counters are recorded with their actual errors.
    """

    def __init__(self, base: Any, args: argparse.Namespace):
        try:
            import pynvml
        except ImportError as error:
            raise RuntimeError(
                "Install nvidia-ml-py for required NVML memory/counter logging"
            ) from error
        self.nvml = pynvml
        self.base = base
        self.interval = args.nvml_interval_seconds
        self.nvml.nvmlInit()
        self.mapping: dict[str, Any] = {}
        properties = base.torch.cuda.get_device_properties(0)
        uuid = getattr(properties, "uuid", None)
        pci = getattr(base.wp.get_device("cuda:0"), "pci_bus_id", None)
        if isinstance(uuid, bytes):
            uuid = uuid.decode("ascii")
        if isinstance(pci, bytes):
            pci = pci.decode("ascii")
        if args.nvml_device_index is not None:
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(args.nvml_device_index)
            self.mapping = {"method": "explicit_physical_index", "index": args.nvml_device_index}
        elif uuid:
            self.handle = pynvml.nvmlDeviceGetHandleByUUID(str(uuid))
            self.mapping = {"method": "torch_cuda_uuid", "uuid": str(uuid)}
        elif pci:
            self.handle = pynvml.nvmlDeviceGetHandleByPciBusId(str(pci))
            self.mapping = {"method": "warp_cuda_pci_bus_id", "pci_bus_id": str(pci)}
        else:
            raise RuntimeError("Cannot map CUDA 0 to NVML; pass its physical --nvml-device-index")
        self.samples: list[dict[str, Any]] = []
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None

    def sample(self) -> dict[str, Any]:
        n = self.nvml
        result: dict[str, Any] = {"measured_at": self.base.utc(), "errors": {}}
        calls = {
            "memory_used_bytes": lambda: int(n.nvmlDeviceGetMemoryInfo(self.handle).used),
            "memory_free_bytes": lambda: int(n.nvmlDeviceGetMemoryInfo(self.handle).free),
            "gpu_utilisation_percent": lambda: int(
                n.nvmlDeviceGetUtilizationRates(self.handle).gpu
            ),
            "memory_utilisation_percent": lambda: int(
                n.nvmlDeviceGetUtilizationRates(self.handle).memory
            ),
            "sm_clock_mhz": lambda: int(n.nvmlDeviceGetClockInfo(self.handle, n.NVML_CLOCK_SM)),
            "memory_clock_mhz": lambda: int(
                n.nvmlDeviceGetClockInfo(self.handle, n.NVML_CLOCK_MEM)
            ),
            "power_watts": lambda: n.nvmlDeviceGetPowerUsage(self.handle) / 1000.0,
            "power_limit_watts": lambda: n.nvmlDeviceGetEnforcedPowerLimit(self.handle) / 1000.0,
        }
        for name, call in calls.items():
            try:
                result[name] = call()
            except n.NVMLError as error:
                result["errors"][name] = str(error)
        try:
            processes = n.nvmlDeviceGetComputeRunningProcesses(self.handle)
            result["compute_processes"] = [
                {
                    "pid": int(p.pid),
                    "used_gpu_memory_bytes": int(p.usedGpuMemory)
                    if p.usedGpuMemory
                    not in (None, getattr(n, "NVML_VALUE_NOT_AVAILABLE", 2**64 - 1))
                    else None,
                }
                for p in processes
            ]
            own = [
                p["used_gpu_memory_bytes"]
                for p in result["compute_processes"]
                if p["pid"] == os.getpid() and p["used_gpu_memory_bytes"] is not None
            ]
            if own:
                result["process_memory_used_bytes"] = sum(own)
        except n.NVMLError as error:
            result["errors"]["compute_processes"] = str(error)
        return result

    def start(self) -> None:
        self.samples = [self.sample()]
        self.stop_event.clear()

        def collect() -> None:
            while not self.stop_event.wait(self.interval):
                self.samples.append(self.sample())

        self.thread = threading.Thread(target=collect, name="dpt-nvml-monitor", daemon=True)
        self.thread.start()

    def stop(self) -> dict[str, Any]:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join()
        self.samples.append(self.sample())
        result: dict[str, Any] = {
            "mapping": self.mapping,
            "sampling_interval_seconds": self.interval,
            "peak_semantics": "maximum observed NVML sample; may miss shorter allocation peaks",
            "samples": self.samples,
        }
        for name in ("memory_used_bytes", "process_memory_used_bytes"):
            values = [sample[name] for sample in self.samples if name in sample]
            if values:
                result["sampled_peak_" + name] = max(values)
        return result

    def close(self) -> None:
        self.nvml.nvmlShutdown()


class Measurements:
    """Measure exact warmed operations; subclasses supply inputs and configuration."""

    def save(self) -> None:
        target = self.args.output / "results.json"
        temporary = target.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(self.results, indent=2, allow_nan=False) + "\n")
        temporary.replace(target)

    def measure(
        self,
        label: str,
        operation: Callable[[], Any],
        metadata: dict[str, Any],
        *,
        stream: Any = None,
        check: Callable[[], Any] | None = None,
    ) -> dict[str, Any]:
        stream = self.stream if stream is None else stream
        self.torch.cuda.synchronize()
        started = time.perf_counter()
        operation()
        stream.synchronize()
        first = time.perf_counter() - started
        if check is not None:
            check()
        started = time.perf_counter()
        operation()
        stream.synchronize()
        pilot = time.perf_counter() - started
        if check is not None:
            check()
        repeats = (
            self.args.repeats
            if pilot < self.args.long_threshold_seconds
            else min(self.args.repeats, self.args.long_repeats)
        )
        result: dict[str, Any] = {
            "label": label,
            "configuration": metadata,
            "first_use_in_this_process_seconds": first,
            "pilot_completed_seconds": pilot,
            "first_use_limit": (
                "not an isolated-cache cold start; earlier cases may have compiled shared modules"
            ),
            "repeats_per_round": repeats,
            "rounds": [],
        }
        self.results["measurements"].append(result)
        for index in range(self.args.rounds):
            self.torch.cuda.reset_peak_memory_stats()
            round_result = {
                "round": index,
                "measured_at": self.base.utc(),
                "wall_ms": [],
                "device_ms": [],
            }
            self.monitor.start()
            try:
                for _ in range(repeats):
                    begin = self.torch.cuda.Event(enable_timing=True)
                    end = self.torch.cuda.Event(enable_timing=True)
                    stream.synchronize()
                    started = time.perf_counter()
                    begin.record(stream)
                    with self.torch.cuda.nvtx.range(label):
                        operation()
                    end.record(stream)
                    end.synchronize()
                    round_result["wall_ms"].append((time.perf_counter() - started) * 1000.0)
                    round_result["device_ms"].append(begin.elapsed_time(end))
                    # Complete deferred numerical checks before resetting a
                    # gradient destination for the next independent sample.
                    if check is not None:
                        check()
            finally:
                round_result["nvml"] = self.monitor.stop()
                round_result["torch_peak_allocated_bytes"] = self.torch.cuda.max_memory_allocated()
                round_result["torch_peak_reserved_bytes"] = self.torch.cuda.max_memory_reserved()
                round_result["finished_at"] = self.base.utc()
                result["rounds"].append(round_result)
                self.save()
            print(
                label,
                "round",
                index + 1,
                "samples",
                repeats,
                "median wall ms",
                float(self.np.median(round_result["wall_ms"])),
                "output",
                self.args.output / "results.json",
                flush=True,
            )
        for kind in ("wall", "device"):
            samples = [x for row in result["rounds"] for x in row[kind + "_ms"]]
            result[kind + "_median_ms"] = float(self.np.median(samples))
            result[kind + "_p95_ms"] = float(self.np.percentile(samples, 95))
        self.save()
        return result

    def error(self, actual: Any, reference: Any) -> dict[str, float]:
        delta = self.np.asarray(actual, dtype=self.np.float64) - reference
        return {
            "max_abs": float(self.np.max(self.np.abs(delta))),
            "rmse": float(self.np.sqrt(self.np.mean(delta**2))),
            "relative_l2": float(
                self.np.linalg.norm(delta) / max(self.np.linalg.norm(reference), 1e-300)
            ),
        }
