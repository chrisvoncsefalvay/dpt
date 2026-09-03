"""Report saved Nsight evidence without running a profiler or importing DPT.

Pass --capture LABEL=DIRECTORY for completed fullscan_profiles.py captures,
--before LABEL=DIRECTORY for optional matched earlier NCU captures, and
--diagnostic LABEL=DIRECTORY for controls without a matched comparison.
--timeline LABEL=timeline.json for final fullscan_timeline.py summaries and
--before-timeline LABEL=timeline.json for earlier application traces. A raw CSV
with adjacent metadata may also be decoded as explicitly labelled legacy
evidence. The output directory must be new. Profile durations are diagnostic.
This helper neither imports nor manufactures clean benchmark timings.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import re
import sys
import textwrap
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks import analyse_profiles as decoder

LEVELS = ("L1", "L2", "DRAM")
ROLE_LABELS = {"before": "reference", "final": "DPT", "diagnostic": "diagnostic"}
HOT_NAMES = ("get_forward__locals__forward", "get_vjp__locals__vjp")
PROJECTION_SOURCES = (
    "python/dpt/torch/modules.py",
    "python/dpt/torch/_projection_ops.py",
    "python/dpt/projection.py",
    "python/dpt/kernels/projection.py",
    "python/dpt/kernels/projection_fp32.py",
)
PROBE_RANGES = {"forward": "probe_forward", "backward": "probe_backward"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def ncu_selection(command, config):
    """Accept recorded complete-operation selectors, never a truncated launch set."""
    require("--launch-count" not in command, "A launch-limited capture is not a whole operation")
    include = f"probe_{config['requested_operation']}/"
    require(
        "--nvtx-include" in command and command[command.index("--nvtx-include") + 1] == include,
        "NVTX selection differs",
    )
    if "--range-filter" in command:
        require(command[command.index("--range-filter") + 1] == "yes:1:1", "NVTX range differs")
        return "first_matching_range_in_first_profiler_interval"
    require(
        config.get("ncu_range_selection") == "unique_matching_nvtx_range"
        and config.get("ncu_range_filter") is None
        and config.get("profile_repeats") == 1
        and "--repeats" in command
        and command[command.index("--repeats") + 1] == "1",
        "An unfiltered NVTX capture must contain exactly one measured matching range",
    )
    for flag, expected in (
        ("--nvtx-push-pop-scope", "process"),
        ("--profile-from-start", "off"),
    ):
        require(
            flag in command and command[command.index(flag) + 1] == expected,
            "Unique-range capture scope differs",
        )
    require("--disable-profiler-start-stop" not in command, "Profiler interval differs")
    return "unique_matching_nvtx_range"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path):
    def reject(value):
        raise ValueError(f"Nonfinite JSON value {value}: {path}")

    return json.loads(path.read_text(), parse_constant=reject)


def labelled(value):
    label, separator, path = value.partition("=")
    require(separator and label and path, "Use LABEL=PATH")
    require(re.fullmatch(r"[A-Za-z0-9_.-]+", label), "Labels need letters, digits, '.', '_' or '-'")
    return label, Path(path).resolve()


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def source_sites(path):
    """Count unique disassembled instruction addresses, never executed events."""
    if not path.exists():
        return None
    sites, function = defaultdict(dict), None
    with path.open(newline="") as stream:
        for row in csv.reader(stream):
            if len(row) > 1 and row[0] == "Function Name":
                function = row[1]
            elif function and len(row) > 3 and row[2].startswith("0x"):
                words = row[3].strip().split()
                if words and words[0].startswith("@"):
                    words = words[1:]
                if words:
                    sites[function][row[2]] = words[0]
    return {name: dict(Counter(addresses.values())) for name, addresses in sites.items()}


def instruction_count(profile, opcode):
    counts = profile["predicated_thread_opcode_counts"]
    return counts.get(opcode, 0) if counts else None


def extra_counters(profile, sites):
    integer = profile["memory_integer_counters"]
    local_sectors = decoder.total(
        integer.get("l1tex__t_sectors_pipe_lsu_mem_local_op_" + direction + ".sum")
        for direction in ("ld", "st")
    )
    opcodes = profile["predicated_thread_opcode_counts"]
    precise_rcp = [key for key in opcodes if key.startswith("MUFU.RCP")]
    # NCU commonly reports only MUFU, which also includes non-reciprocal work.
    profile["additional_counters"] = {
        "conversion_thread_instructions": profile["predicated_thread_category_counts"].get(
            "Conversion"
        ),
        "MUFU_thread_instructions": instruction_count(profile, "MUFU"),
        "RCP_thread_instructions": sum(opcodes[key] for key in precise_rcp)
        if precise_rcp
        else None,
        "RCP_count_scope": "exact RCP opcode instances"
        if precise_rcp
        else "not exported separately",
        "local_memory_sector_bytes": decoder.product(local_sectors, 32),
        "static_sass_instruction_sites": sites.get(profile["kernel"]) if sites else None,
        "static_site_limit": (
            "unique instruction addresses in the exported SASS"
        ),
    }


def check_capture_files(path, capture, required_exports):
    """Bind parser inputs. Compact downloads may omit explicitly checked binaries."""
    if capture.get("schema_version", 1) < 2:
        return {
            "status": "legacy_unbound_exports",
            "limit": "capture metadata uses schema version 1 with unbound export identities",
        }
    records = capture["exports"]
    names = [entry["file"] for entry in records]
    require(
        len(names) == len(set(names)) and set(required_exports) <= set(names),
        "Required parser exports are missing or duplicated",
    )
    for entry in records:
        relative = Path(entry["file"])
        require(
            not relative.is_absolute() and ".." not in relative.parts,
            "Export path escapes capture directory",
        )
        actual = path / relative
        require(
            actual.is_file()
            and actual.stat().st_size == entry["bytes"]
            and sha256(actual) == entry["sha256"],
            "Captured export changed: " + str(relative),
        )
    binary_checks = capture["binary_report_checks"]
    require(
        binary_checks["passed"] is True
        and binary_checks["files"] == capture["reports"]
        and binary_checks["checked_at_utc"],
        "Missing capture-time binary hash check",
    )
    binary_status = []
    for entry in capture["reports"]:
        relative = Path(entry["file"])
        require(
            not relative.is_absolute() and ".." not in relative.parts,
            "Binary path escapes capture directory",
        )
        actual = path / relative
        if actual.exists():
            require(
                actual.is_file()
                and actual.stat().st_size == entry["bytes"]
                and sha256(actual) == entry["sha256"],
                "Captured binary changed",
            )
        binary_status.append(
            {
                "file": entry["file"],
                "sha256": entry["sha256"],
                "verified_locally": actual.exists(),
                "capture_time_check": binary_checks["checked_at_utc"],
            }
        )
    require(binary_status, "No captured binary identity")
    return {"status": "verified_exports", "exports": records, "binaries": binary_status}


def ncu_raw_export(capture, command):
    """Resolve the recorded output stem and its matching raw report import."""
    require(
        command.count("-o") == 1 and command.index("-o") + 1 < len(command),
        "NCU capture must record exactly one output stem",
    )
    stem = Path(command[command.index("-o") + 1])
    require(stem.name and ".." not in stem.parts, "Invalid NCU output stem")
    reports = capture["reports"]
    require(
        len(reports) == 1
        and reports[0]["file"] in (stem.name + ".ncu-rep", stem.name + ".ncu-repz"),
        "Captured binary does not match the NCU output stem",
    )
    name = stem.name + ".csv"
    exports = [row for row in capture["commands"] if row["stdout"] == name]
    require(len(exports) == 1, "Missing or duplicate raw NCU export command")
    exported = exports[0]["command"]
    for flag, expected in (
        ("--import", str(stem.parent / reports[0]["file"])),
        ("--page", "raw"),
    ):
        require(
            exported.count(flag) == 1
            and exported.index(flag) + 1 < len(exported)
            and exported[exported.index(flag) + 1] == expected,
            "Raw NCU export differs: " + flag,
        )
    require(
        "--csv" in exported and "--print-metric-instances" not in exported,
        "Raw NCU export is not the complete counter CSV",
    )
    return name


def gradient_binding(config, probe, command):
    """Validate explicit derivative metadata or retain a declared older joint route."""
    keys = ("gradients", "requested_gradients", "pose_requires_grad")
    markers = (
        *keys,
        "gradient_shapes",
        "gradient_dtypes",
        "pose_representation",
        "target_source_files",
        "nvtx_ranges",
        "current_torch_stream",
    )
    explicit = any(key in record for record in (config, probe) for key in markers)
    if not explicit:
        declared_joint = config.get("backward") == "joint torch.autograd.grad for field and pose"
        require("--gradients" not in command, "A gradient-selectable probe needs explicit metadata")
        return {
            "gradients": "both" if declared_joint else None,
            "requested_gradients": ["volume", "pose"] if declared_joint else None,
            "pose_requires_grad": True if declared_joint else None,
            "gradient_metadata": "legacy_declared_joint" if declared_joint else "unrecorded",
        }
    require(
        all(key in record for record in (config, probe) for key in keys),
        "Incomplete requested-gradient metadata",
    )
    mode = config["gradients"]
    require(mode in ("volume", "both"), "Unknown requested gradients")
    requested = ["volume", "pose"] if mode == "both" else ["volume"]
    require(
        probe["gradients"] == mode
        and config["requested_gradients"] == probe["requested_gradients"] == requested
        and config["pose_requires_grad"] is (mode == "both")
        and probe["pose_requires_grad"] is (mode == "both"),
        "Probe and capture requested gradients differ",
    )
    require(
        command.count("--gradients") == 1 and command[command.index("--gradients") + 1] == mode,
        "Captured gradient command differs",
    )
    representation = (
        "shared six-component right tangent at identity"
        if mode == "both"
        else "fixed identity geometry without a pose tensor"
    )
    require(probe["pose_representation"] == representation, "Pose representation differs")
    return {
        "gradients": mode,
        "requested_gradients": requested,
        "pose_requires_grad": mode == "both",
        "gradient_metadata": "explicit",
    }


def probe_operation_digest(path):
    """Compare saved execution functions while retaining exact file hashes separately."""
    tree = ast.parse(path.read_text())
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in ("load_helper", "run")
    ]
    require(
        {node.name for node in functions} == {"load_helper", "run"},
        "Captured probe lacks its execution functions",
    )
    for function in functions:
        for node in ast.walk(function):
            if not (
                isinstance(node, ast.Assign)
                and isinstance(node.value, ast.Dict)
                and any(
                    isinstance(target, ast.Name) and target.id == "result"
                    for target in node.targets
                )
            ):
                continue
            value = node.value
            for index in reversed(range(len(value.keys))):
                key = value.keys[index]
                if isinstance(key, ast.Constant) and key.value == "derived_from_probe_sha256":
                    del value.keys[index]
                    del value.values[index]
    return hashlib.sha256("\n".join(ast.dump(node) for node in functions).encode()).hexdigest()


def validate_probe_binding(path, capture, command):
    """Bind saved source, helper, shape and derivative facts before decoding counters."""
    config, probe, inputs = capture["configuration"], capture["probe_result"], capture["input"]
    source = capture["source"]
    require(
        probe["finite_outputs"] is True
        and probe["source_unchanged"] is True
        and probe["source"] == source
        and probe["precision"] == config["precision"]
        and probe["views"] == config["views"]
        and probe["view_indices"] == config["view_indices"]
        and probe["cotangent_sha256"] == inputs["cotangent_sha256"],
        "Probe source, inputs or completion differs",
    )
    files = source.get("files")
    require(isinstance(files, dict) and files, "Missing source file hashes")
    require(
        all(
            isinstance(name, str)
            and not Path(name).is_absolute()
            and ".." not in Path(name).parts
            and isinstance(value, str)
            and re.fullmatch(r"[0-9a-f]{64}", value)
            for name, value in files.items()
        ),
        "Invalid source file hashes",
    )
    require(
        hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest() == source["sha256"],
        "Source aggregate differs from file hashes",
    )
    require(probe["input_metadata"] == inputs["metadata"], "Probe input metadata differs")
    require(load_json(path / "probe/probe.json") == probe, "Saved probe report differs")
    snapshot = capture["probe_snapshot"]
    relative = Path(snapshot["file"])
    require(not relative.is_absolute() and ".." not in relative.parts, "Probe path escapes")
    require(
        snapshot["sha256"] == capture["helpers"]["probe"]["sha256"], "Probe helper hash differs"
    )
    require(probe["harness_sha256"] == snapshot["sha256"], "Probe execution hash differs")
    require(
        (path / relative).is_file() and sha256(path / relative) == snapshot["sha256"],
        "Captured probe copy changed or is missing",
    )
    require(
        probe["helper_sha256"]
        == {
            "fullscan_inputs.py": capture["helpers"]["input_helper"]["sha256"],
            "common.py": capture["helpers"]["measurement_common"]["sha256"],
        },
        "Probe input or measurement helper differs",
    )
    gradients = gradient_binding(config, probe, command)
    for flag, expected in (
        ("--precision", config["precision"]),
        ("--views", str(config["views"])),
        ("--source", capture["source_path"]),
    ):
        require(
            command.count(flag) == 1 and command[command.index(flag) + 1] == expected,
            "Capture command differs: " + flag,
        )
    gradients["probe_operation_sha256"] = probe_operation_digest(path / relative)
    if gradients["gradient_metadata"] == "explicit":
        shape = inputs["metadata"]["arrays"]["attenuation"]["shape"]
        require(
            shape == list(reversed(inputs["metadata"]["grid"]["shape"])),
            "Field array shape differs from grid axes",
        )
        expected_shapes = [shape] + ([[6]] if gradients["gradients"] == "both" else [])
        expected_dtypes = ["torch.float32"] + (
            ["torch.float64"] if gradients["gradients"] == "both" else []
        )
        require(
            probe["gradient_shapes"] == expected_shapes
            and probe["gradient_dtypes"] == expected_dtypes,
            "Gradient shapes or dtypes differ",
        )
        require(
            probe["target_source_files"] == {name: files[name] for name in PROJECTION_SOURCES},
            "Target projection sources differ",
        )
        require(
            probe["nvtx_ranges"] == config["nvtx_ranges"] == PROBE_RANGES, "Probe ranges differ"
        )
        samples = inputs["metadata"]["sampling"]["samples_per_ray"]
        require(
            probe["sampling"] == {"rule": "clipped midpoint", "samples_per_ray": samples},
            "Probe sampling differs",
        )
        require(
            type(probe["current_torch_stream"]) is int and probe["current_torch_stream"] >= 0,
            "Invalid captured Torch stream",
        )
    indices = probe["view_indices"]
    acquisition = inputs["metadata"]["acquisition"]
    require(
        isinstance(indices, list)
        and len(indices) == config["views"]
        and len(set(indices)) == len(indices)
        and all(type(index) is int and 0 <= index < acquisition["views"] for index in indices),
        "Invalid captured view indices",
    )
    return gradients


def load_capture(label, path, role):
    legacy = path.is_file()
    if legacy:
        raw = path
        metadata_path = raw.with_suffix(".metadata.json")
        require(
            metadata_path.is_file(), f"Legacy raw CSV needs its actual metadata: {metadata_path}"
        )
        capture = load_json(metadata_path)
        require(capture.get("completed_checks_passed") is True, "Incomplete legacy capture")
        grid = capture["input"]["grid"]
        panels = capture["resolved_geometry"]
        workload = {
            "legacy": True,
            "precision": capture["precision"],
            "operation": capture["operation"],
            "views": len(panels),
            "view_labels": capture["views"],
            "grid": grid,
            "detector_shape": panels[0]["shape"],
            "samples_per_ray": capture["native_spec"]["samples_per_ray"],
            "resampled": capture["input"].get("resampled"),
            "scope": "native-grid capture: " + capture["boundary"],
        }
        completed = capture["profile_completed_at"]
        signature, command = None, None
        integrity = {"status": "legacy_unbound_exports"}
        gradients = {
            "gradients": None,
            "requested_gradients": None,
            "pose_requires_grad": None,
            "gradient_metadata": "unrecorded",
        }
    else:
        metadata_path = path / "capture.json"
        capture = load_json(metadata_path)
        require(capture.get("status") == "complete", f"Capture incomplete: {path}")
        for key in ("source_unchanged", "helpers_unchanged", "probe_snapshot_unchanged"):
            require(capture.get(key) is True, f"Capture binding failed: {key}")
        config, inputs, probe = capture["configuration"], capture["input"], capture["probe_result"]
        require(config["profiler"] == "ncu", "Pass analysed NSYS JSON through --timeline")
        require(
            probe["finite_outputs"] is True
            and probe["source_unchanged"] is True
            and probe["source"] == capture["source"]
            and probe["view_indices"] == config["view_indices"]
            and probe["precision"] == config["precision"]
            and probe["views"] == config["views"]
            and probe["cotangent_sha256"] == inputs["cotangent_sha256"]
            and probe["harness_sha256"] == capture["probe_snapshot"]["sha256"],
            "Probe source, seed, views or harness differs from capture",
        )
        require(
            inputs["cotangent_metadata"]["sha256"] == inputs["cotangent_sha256"]
            and inputs["cotangent_metadata"]["contract_sha256"]
            == capture["frozen_contract_sha256"],
            "Cotangent does not belong to the recorded contract",
        )
        require(
            all(row.get("returncode") == 0 for row in capture["commands"]),
            "A profiler/export command did not complete successfully",
        )
        command = next(
            row["command"] for row in capture["commands"] if row["stdout"] == "profiler-stdout.txt"
        )
        gradients = validate_probe_binding(path, capture, command)
        selection = ncu_selection(command, config)
        raw = path / ncu_raw_export(capture, command)
        integrity = check_capture_files(
            path,
            capture,
            [raw.name, raw.stem + "-instances.csv", raw.stem + "-source.csv", "probe/probe.json"],
        )
        sidecar = raw.with_suffix(".metadata.json")
        if sidecar.exists():
            require(load_json(sidecar) == capture, "Capture metadata sidecar differs")
        info = inputs["metadata"]
        workload = {
            "legacy": False,
            "precision": config["precision"],
            "operation": config["requested_operation"],
            "views": config["views"],
            "view_indices": config["view_indices"],
            "grid": info["grid"],
            "detector_shape": info["acquisition"]["detector_shape"],
            "samples_per_ray": 1024,
            "sampling": config["sampling"],
            "pose": config["pose"],
            "backward": config["backward"],
            "scope": config["capture_scope"],
            "input_metadata_sha256": inputs["metadata_sha256"],
            "cotangent_sha256": inputs["cotangent_sha256"],
            "frozen_contract_sha256": capture["frozen_contract_sha256"],
            **gradients,
        }
        signature = {
            "workload": workload,
            "profiling": {
                key: config[key]
                for key in (
                    "ncu_replay_mode",
                    "ncu_cache_control",
                    "ncu_clock_control",
                    "profile_repeats",
                )
            },
            "helpers": {key: value["sha256"] for key, value in capture["helpers"].items()},
            "versions": probe["versions"],
            "torch_cuda": probe["torch_cuda"],
        }
        signature["profiling"]["nvtx_scope"] = config.get("nvtx_scope")
        signature["profiling"]["range_selection"] = selection
        signature["profiling"]["sections"] = [
            command[index + 1] for index, word in enumerate(command) if word == "--section"
        ]
        signature["profiling"]["metrics"] = [
            command[index + 1] for index, word in enumerate(command) if word == "--metrics"
        ]
        signature["helpers"] = {
            key: capture["helpers"][key]["sha256"] for key in ("input_helper", "measurement_common")
        }
        if integrity["status"] != "verified_exports":
            signature = None
        if gradients["gradient_metadata"] == "unrecorded":
            signature = None
        completed = capture["completed_at_utc"]
    profiles = decoder.analyse(raw)
    require(profiles, "Capture contains no decoded kernels")
    sites_path = raw.with_name(raw.stem + "-source.csv")
    sites = source_sites(sites_path)
    for profile in profiles:
        extra_counters(profile, sites)
        profile.update(
            id=f"{label}:k{profile['ncu_kernel_id']}",
            operation_id=label,
            role=role,
            plotted_hot_kernel=any(name in profile["kernel"] for name in HOT_NAMES),
            requested_gradients=gradients["requested_gradients"],
        )
    summary = decoder.operation_summary(raw.stem, profiles, raw.parent)
    summary.pop("capture_metadata")
    summary.update(
        id=label,
        role=role,
        scope=workload["scope"],
        workload=workload,
        source_sha256=capture["source"]["sha256"],
        source=capture["source"],
        capture_helpers=capture.get("helpers"),
        measurement_completed_at=completed,
        capture_metadata={"file": str(metadata_path), "sha256": sha256(metadata_path)},
        raw_csv={"file": str(raw), "sha256": sha256(raw)},
        source_export={"file": str(sites_path), "sha256": sha256(sites_path)}
        if sites_path.exists()
        else None,
        capture_command=command,
        reports=capture.get("reports"),
        comparison_signature=signature,
        file_integrity=integrity,
    )
    return summary, profiles


def load_timeline(label, path, role="final"):
    result = load_json(path)
    checks = result["checks"]
    require(
        checks["all_gpu_events_attributed"] is True
        and checks["capture_kernel_count"] == checks["summed_range_kernel_count"]
        and checks["capture_kernel_sum_ns"] == checks["summed_range_kernel_sum_ns"],
        "Timeline has incomplete kernel attribution",
    )
    explicit = any(
        key in result for key in ("gradients", "requested_gradients", "pose_requires_grad")
    )
    if explicit:
        mode = result.get("gradients")
        targets = ["volume", "pose"] if mode == "both" else ["volume"]
        require(
            mode in ("volume", "both")
            and result.get("requested_gradients") == targets
            and result.get("pose_requires_grad") is (mode == "both"),
            "Timeline derivative metadata is inconsistent",
        )
    else:
        declared_joint = result["probe_workload"].startswith("both field/pose active")
        mode = "both" if declared_joint else None
        targets = ["volume", "pose"] if declared_joint else None
    return {
        "id": label,
        "role": role,
        "file": str(path),
        "sha256": sha256(path),
        "source_sha256": result["source_sha256"],
        "precision": result["precision"],
        "views": result["views"],
        "view_indices": result.get("view_indices"),
        "input_metadata": result.get("input_metadata"),
        "measurement_started_at": result["measurement_started_at"],
        "measurement_completed_at": result["measurement_completed_at"],
        "probe_workload": result["probe_workload"],
        "probe_status_counts": result.get("probe_status_counts"),
        "gradients": mode,
        "requested_gradients": targets,
        "pose_requires_grad": mode == "both" if mode is not None else None,
        "cotangent": result.get("cotangent"),
        "profile_harness_sha256": result.get("profile_harness_sha256"),
        "sqlite": result["sqlite"],
        "metadata": result["metadata"],
        "ranges": result["ranges"],
        "checks": checks,
        "method": result["method"],
        "attribution_scope": result.get("attribution_scope"),
    }


def timeline_bindings(trace):
    """Keep an incomplete archive usable without inventing matched inputs."""
    metadata = trace["input_metadata"]
    cotangent = trace["cotangent"]
    bindings = {
        "input_metadata": metadata,
        "view_indices": trace["view_indices"],
        "views": trace["views"],
        "precision": trace["precision"],
        "cotangent_sha256": cotangent.get("sha256") if isinstance(cotangent, dict) else None,
        "profile_harness_sha256": trace["profile_harness_sha256"],
        "sampling": metadata.get("sampling") if isinstance(metadata, dict) else None,
        "probe_workload": trace["probe_workload"],
        "requested_gradients": trace["requested_gradients"],
        "pose_requires_grad": trace["pose_requires_grad"],
        "ranges": [(row["range"], row["occurrence"]) for row in trace["ranges"]],
    }
    missing = [key for key, value in bindings.items() if value is None or value == ""]
    indices = bindings["view_indices"]
    if indices is not None and (
        not isinstance(indices, list)
        or len(indices) != trace["views"]
        or any(type(index) is not int or index < 0 for index in indices)
    ):
        missing.append("valid_explicit_view_indices")
    for key in ("input_metadata", "sampling"):
        if bindings[key] is not None and not isinstance(bindings[key], dict):
            missing.append(f"valid_{key}")
        elif bindings[key] == {}:
            missing.append(key)
    if not bindings["ranges"]:
        missing.append("ranges")
    return bindings, sorted(set(missing))


def compare_timelines(traces):
    """Identify exact workload pairs. Unbound archives retain only raw evidence."""
    final = [trace for trace in traces if trace["role"] == "final"]
    comparisons = []
    for before in (trace for trace in traces if trace["role"] == "before"):
        before_bindings, before_missing = timeline_bindings(before)
        matches, candidates = [], []
        for after in final:
            after_bindings, after_missing = timeline_bindings(after)
            differences = [
                key for key, value in before_bindings.items() if value != after_bindings[key]
            ]
            candidates.append(
                {
                    "after": after["id"],
                    "different_bindings": differences,
                    "missing_before_bindings": before_missing,
                    "missing_after_bindings": after_missing,
                }
            )
            if not differences and not before_missing and not after_missing:
                matches.append(after["id"])
        matched = len(matches) == 1
        comparisons.append(
            {
                "before": before["id"],
                "after": matches[0] if matched else None,
                "status": "matched" if matched else "unmatched",
                "reason": (
                    "Exactly one final trace has complete, identical workload bindings"
                    if matched
                    else "No unique final trace has complete, identical workload bindings"
                ),
                "required_bindings": list(before_bindings),
                "candidates": candidates,
            }
        )
    return comparisons


def aggregate(profiles):
    def summed(access):
        return decoder.total(access(profile) for profile in profiles)

    result = {
        "kernels": len(profiles),
        "FP32_FLOPs": summed(lambda p: p["precision"]["fp32"]["add_mul_fma_flops"]),
        "FP64_FLOPs": summed(lambda p: p["precision"]["fp64"]["add_mul_fma_flops"]),
        "conversion_thread_instructions": summed(
            lambda p: p["additional_counters"]["conversion_thread_instructions"]
        ),
        "MUFU_thread_instructions": summed(
            lambda p: p["additional_counters"]["MUFU_thread_instructions"]
        ),
        "RCP_thread_instructions": summed(
            lambda p: p["additional_counters"]["RCP_thread_instructions"]
        ),
        "register_spill_instructions": summed(lambda p: p["register_spill_instructions"]),
        "local_memory_sector_bytes": summed(
            lambda p: p["additional_counters"]["local_memory_sector_bytes"]
        ),
    }
    for opcode in ("F2F", "F2I", "I2F", "UI2F", "ATOM", "ATOMG", "RED", "REDG"):
        result[opcode + "_thread_instructions"] = summed(
            lambda p, opcode=opcode: instruction_count(p, opcode)
        )
    for level in LEVELS:
        result[level + "_counter_bytes"] = summed(
            lambda p, level=level: p["memory_counter_bytes"][level]
        )
    registers = [p["registers_per_thread"] for p in profiles]
    result["max_registers_per_thread"] = (
        max(registers) if registers and all(x is not None for x in registers) else None
    )
    return result


def compare_captures(operations, profiles):
    comparisons = []
    for before in (op for op in operations if op["role"] == "before"):
        require(
            before["comparison_signature"] is not None,
            "Before/after needs complete capture metadata",
        )
        matches = [
            op
            for op in operations
            if op["role"] == "final"
            and op["comparison_signature"] == before["comparison_signature"]
        ]
        require(
            len(matches) == 1, f"Expected one identical-workload final capture for {before['id']}"
        )
        after = matches[0]
        for scope in ("whole_operation", "projector_and_vjp_kernels"):
            groups = [
                [
                    p
                    for p in profiles
                    if p["operation_id"] == operation["id"]
                    and (scope == "whole_operation" or p["plotted_hot_kernel"])
                ]
                for operation in (before, after)
            ]
            require(
                {(p["device"], p["compute_capability"], p["sm_count"]) for p in groups[0]}
                == {(p["device"], p["compute_capability"], p["sm_count"]) for p in groups[1]},
                "Before/after profiling devices differ",
            )
            values = [aggregate(group) for group in groups]
            for metric in values[0]:
                a, b = values[0][metric], values[1][metric]
                delta = b - a if a is not None and b is not None else None
                comparisons.append(
                    {
                        "before": before["id"],
                        "after": after["id"],
                        "scope": scope,
                        "metric": metric,
                        "before_value": a,
                        "after_value": b,
                        "delta": delta,
                        "change_percent": 100 * delta / a if delta is not None and a else None,
                    }
                )
    return comparisons


def counters_rows(profiles):
    rows = []
    for p in profiles:
        row = {
            "capture": p["operation_id"],
            "role": p["role"],
            "kernel_id": p["ncu_kernel_id"],
            "kernel": p["kernel"],
            "plot_selected": p["plotted_hot_kernel"],
            "requested_gradients": "+".join(p["requested_gradients"])
            if p.get("requested_gradients") is not None
            else None,
            "replay_kernel_ms": decoder.product(p["profiled_kernel_seconds"], 1000),
            "registers_per_thread": p["registers_per_thread"],
            "occupancy_achieved_percent": p["occupancy_achieved_percent"],
            "FP64_pipe_active_percent": p["fp64_pipe_active_percent"],
            **aggregate([p]),
            **{
                f"{precision.upper()}_FLOPs_per_L2_requested_sector_byte": p["precision"][
                    precision
                ]["arithmetic_intensity_counter_flops_per_byte"]["L2"]
                for precision in ("fp32", "fp64")
            },
        }
        row.update(
            {
                "throughput_" + key + "_percent": value
                for key, value in p["throughput_percent_of_peak"].items()
            }
        )
        row.update(
            {
                "stall_" + key + "_cycles_per_issued_instruction": value
                for key, value in p["stall_warp_cycles_per_issued_warp_instruction"].items()
            }
        )
        rows.append(row)
    return rows


def roofline(profiles, operations, precision, output):
    """Plot only matching-precision projector/VJP kernels. Retain all rows elsewhere."""
    operations = {op["id"]: op for op in operations}
    selected = [
        p
        for p in profiles
        if p["plotted_hot_kernel"]
        and operations[p["operation_id"]]["workload"]["precision"] == precision
        and operations[p["operation_id"]]["role"] == "final"
    ]
    points = [
        (
            p,
            level,
            p["precision"][precision]["arithmetic_intensity_flops_per_byte"][level],
            p["precision"][precision]["flops_per_second"],
        )
        for p in selected
        for level in LEVELS
    ]
    points = [
        (p, level, x, y / 1e9)
        for p, level, x, y in points
        if x is not None and y is not None and x > 0 and y > 0
    ]
    ceilings = [
        p
        for p in selected
        if p["precision"][precision]["model_peak_flops_per_second"]
        and all(p["bandwidth_model_peak_bytes_per_second"].values())
    ]
    if not points or not ceilings:
        return {
            "created": False,
            "reason": "Measured positive points or complete theoretical ceilings unavailable",
        }
    require(len({p["device"] for p in selected}) == 1, "Use separate reports for different devices")
    # Draw every captured clock-dependent roof. No unmeasured maximum or averaged clock.
    plt, np = decoder.plt, decoder.np
    fig, ax = plt.subplots(figsize=(9.2, 6.2))
    low = 10 ** np.floor(np.log10(min(x for _, _, x, _ in points)) - 0.3)
    high = max(1e3, 10 ** np.ceil(np.log10(max(x for _, _, x, _ in points)) + 0.3))
    xaxis = np.geomspace(low, high, 400)
    for p in ceilings:
        peak = p["precision"][precision]["model_peak_flops_per_second"] / 1e9
        for level, style in (("L1", ":"), ("L2", "--"), ("DRAM", "-")):
            ax.plot(
                xaxis,
                np.minimum(peak, xaxis * p["bandwidth_model_peak_bytes_per_second"][level] / 1e9),
                color="black",
                linestyle=style,
                linewidth=0.7,
                alpha=0.45,
            )
    for index, p in enumerate(selected):
        values = [(level, x, y) for row, level, x, y in points if row["id"] == p["id"]]
        if not values:
            continue
        color = decoder.COLORS[index % len(decoder.COLORS)]
        operation = operations[p["operation_id"]]
        label = (
            f"{operation['id']} · {operation_label(operation['workload'])} · "
            f"{operation['workload']['views']} view(s)"
        )
        ax.plot(
            [x for _, x, _ in values],
            [y for _, _, y in values],
            color=color,
            linewidth=0.8,
            label=label,
        )
        for level, x, y in values:
            ax.scatter(
                x, y, color=color, marker={"L1": "o", "L2": "^", "DRAM": "s"}[level], s=35, zorder=3
            )
    ax.set(
        xscale="log",
        yscale="log",
        xlim=(low, high),
        xlabel=f"{precision.upper()} arithmetic intensity (FLOP/byte)",
        ylabel=f"{precision.upper()} throughput (GFLOP/s)",
    )
    ax.set_title(f"{precision.upper()} projector/VJP kernels", loc="left", fontsize=12)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(which="major", color="#e5e5e5", linewidth=0.5)
    ax.legend(
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=1,
        fontsize=7,
        loc="upper left",
    )
    scopes = list(
        dict.fromkeys(workload_label(operations[p["operation_id"]]["workload"]) for p in selected)
    )
    caption_parts = [
        selected[0]["device"],
        *scopes,
        "Theoretical NVIDIA ceilings at each capture's recorded clocks.",
        "L1 dotted/circle · L2 dashed/triangle · DRAM solid/square. "
        "L1/L2 use outgoing writeback proxies.",
        "L2 writeback measures outgoing traffic. "
        "Total requested-sector counters include reductions.",
        "All captured kernels and total requested-sector bytes remain in the numeric results.",
    ]
    caption = "\n".join(textwrap.fill(part, width=118) for part in caption_parts)
    bottom_inches = max(6.2 * 0.29, 0.8 + 0.2 * len(caption.splitlines()))
    height_inches = 6.2 + bottom_inches - 6.2 * 0.29
    fig.set_size_inches(9.2, height_inches)
    fig.text(
        0.12, 0.15 / height_inches, caption, ha="left", va="bottom", fontsize=8, linespacing=1.25
    )
    fig.subplots_adjust(
        bottom=bottom_inches / height_inches,
        left=0.12,
        right=0.98,
        top=1 - (6.2 * 0.09) / height_inches,
    )
    files = []
    for suffix in ("svg", "pdf", "png"):
        path = output / f"roofline_{precision}.{suffix}"
        fig.savefig(path, dpi=300, facecolor="white")
        files.append({"file": path.name, "sha256": sha256(path)})
    plt.close(fig)
    return {
        "created": True,
        "plotted_kernel_ids": [p["id"] for p in selected],
        "ceiling_kernel_ids": [p["id"] for p in ceilings],
        "files": files,
    }


def workload_label(workload):
    grid = "x".join(map(str, workload["grid"]["shape"]))
    panel = "x".join(map(str, workload["detector_shape"]))
    view = f"{workload['views']} views"
    if workload.get("view_labels") == ["ap"]:
        view = "1 AP view"
    elif workload.get("view_indices") == [0]:
        view = "1 zero-angle view (saved index 0)"
    elif workload["views"] == 1 and len(workload.get("view_indices", [])) == 1:
        view = f"1 saved view (index {workload['view_indices'][0]})"
    qualifier = "native-grid capture" if workload["legacy"] else "full-extent resampled chest"
    return f"{qualifier}: {grid}, {view}, {panel}, {workload['samples_per_ray']} midpoints"


def operation_label(workload):
    if workload["operation"] == "forward":
        return "forward"
    gradients = workload.get("requested_gradients")
    if gradients == ["volume"]:
        return "volume VJP"
    if gradients == ["volume", "pose"]:
        return "joint VJP"
    return "backward, gradients unrecorded"


def formatted(value, digits=3):
    return "unavailable" if value is None else f"{value:,.{digits}f}"


def table_row(*values):
    return "| " + " | ".join(map(str, values)) + " |"


def markdown(result):
    operations = [op for op in result["operations"] if op["role"] == "final"]
    profiles = [p for p in result["profiles"] if p["role"] == "final"]
    timelines = [t for t in result["timelines"] if t["role"] == "final"]
    lines = [
        "# Fullscan profiling evidence",
        "",
        "Nsight Compute durations are replay-profiled kernel sums. Nsight Systems "
        "ranges are instrumented intervals. "
        "Completed-call medians are reported in the timing tables.",
        "",
        "FLOPs count predicated-thread ADD + MUL + 2 * FMA. "
        "Missing counters are marked unavailable.",
        "",
        "## Nsight Compute capture scope",
        "",
    ]
    for op in operations:
        lines.append(
            f"- **{op['id']}**: {workload_label(op['workload'])}. "
            f"{operation_label(op['workload'])}. {op['scope']}. "
            f"Completed `{op['measurement_completed_at']}`."
        )
    lines += [
        "",
        "Roofline figures use the named projector/VJP kernels and theoretical NVIDIA "
        "ceilings at their recorded clocks. The numeric report contains every captured "
        "kernel, input/source digest, command and report hash.",
        "",
        "| Capture | Kernels | Replay kernel sum (ms) | FP32 FLOPs | FP64 FLOPs |",
        "|---|---:|---:|---:|---:|",
    ]
    for op in operations:
        lines.append(
            table_row(
                op["id"],
                op["kernel_count"],
                formatted(decoder.product(op["sum_profiled_kernel_seconds"], 1000)),
                formatted(op["flops"]["fp32"], 0),
                formatted(op["flops"]["fp64"], 0),
            )
        )
    lines += [
        "",
        "| Projector/VJP kernel | Registers/thread | Occupancy (%) | "
        "FP64 pipe active (%) | Spills | Local sector bytes |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for p in profiles:
        if p["plotted_hot_kernel"]:
            lines.append(
                table_row(
                    p["id"],
                    formatted(p["registers_per_thread"], 0),
                    formatted(p["occupancy_achieved_percent"], 2),
                    formatted(p["fp64_pipe_active_percent"], 2),
                    formatted(p["register_spill_instructions"], 0),
                    formatted(p["additional_counters"]["local_memory_sector_bytes"], 0),
                )
            )
    lines += [
        "",
        "The rooflines use outgoing L1/L2 writeback traffic. The table below reports "
        "total L2 requested-sector bytes, `lts__t_bytes.sum = 32 * lts__t_sectors.sum`, "
        "including atomic and reduction requests. "
        "Each intensity uses the byte count in its column.",
        "",
        "| Projector/VJP kernel | Total L2 requested-sector bytes | "
        "FP32 FLOP/L2 requested-sector byte | FP64 FLOP/L2 requested-sector byte |",
        "|---|---:|---:|---:|",
    ]
    for p in profiles:
        if p["plotted_hot_kernel"]:
            lines.append(
                table_row(
                    p["id"],
                    formatted(p["memory_counter_bytes"]["L2"], 0),
                    *[
                        formatted(
                            p["precision"][precision][
                                "arithmetic_intensity_counter_flops_per_byte"
                            ]["L2"],
                            6,
                        )
                        for precision in ("fp32", "fp64")
                    ],
                )
            )
    if timelines:
        lines += [
            "",
            "## Nsight Systems application ranges",
            "",
            "NVTX intervals include CPU execution and waits. Kernel sums report GPU execution. "
            "These intervals overlap. "
            "Trace gaps measure intervals with no attributed GPU activity.",
            "",
            "| Capture | Precision | Gradients | Views | Range | NVTX (ms) | "
            "Kernels | Kernel sum (ms) | Volume scans | Scan time (ms) | 4-byte D2H | "
            "Sync APIs | Trace gaps (ms) |",
            "|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for trace in timelines:
            for row in trace["ranges"]:
                lines.append(
                    table_row(
                        trace["id"],
                        trace["precision"],
                        "+".join(trace["requested_gradients"])
                        if trace["requested_gradients"] is not None
                        else "unrecorded",
                        trace["views"],
                        f"{row['range']} #{row['occurrence']}",
                        formatted(row["inclusive_nvtx_ms"]),
                        row["kernel_count"],
                        formatted(row["kernel_sum_ms"]),
                        row["full_volume_validation_count"],
                        formatted(row["full_volume_validation_sum_ms"]),
                        row["four_byte_d2h_count"],
                        row["synchronise_api_count"],
                        formatted(row["no_attributed_gpu_activity_inside_nvtx_ms"]),
                    )
                )
        for trace in timelines:
            workload = trace["probe_workload"].replace("; ", ". ")
            workload = re.sub(
                r"Seed: (\S+)", lambda match: f"Seed: `data/{Path(match.group(1)).name}`", workload
            )
            lines += [
                "",
                f"`{trace['id']}`: {workload}. Completed `{trace['measurement_completed_at']}`.",
            ]
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--capture", action="append", type=labelled, default=[], metavar="LABEL=PATH"
    )
    parser.add_argument(
        "--before", action="append", type=labelled, default=[], metavar="LABEL=PATH"
    )
    parser.add_argument(
        "--diagnostic",
        action="append",
        type=labelled,
        default=[],
        metavar="LABEL=PATH",
        help="Measured control with independent source/input identity",
    )
    parser.add_argument(
        "--timeline", action="append", type=labelled, default=[], metavar="LABEL=JSON"
    )
    parser.add_argument(
        "--before-timeline", action="append", type=labelled, default=[], metavar="LABEL=JSON"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--font", type=Path, help="Geist-Regular.ttf. Required when producing plots"
    )
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)
    require(
        args.capture or args.diagnostic or args.timeline or args.before_timeline,
        "Supply actual captures or timeline analyses",
    )
    labels = [
        label
        for label, _ in args.capture
        + args.before
        + args.diagnostic
        + args.timeline
        + args.before_timeline
    ]
    require(len(labels) == len(set(labels)), "Evidence labels must be unique")
    operations, profiles = [], []
    for role, entries in (
        ("final", args.capture),
        ("before", args.before),
        ("diagnostic", args.diagnostic),
    ):
        for label, path in entries:
            operation, records = load_capture(label, path, role)
            operations.append(operation)
            profiles.extend(records)
    traces = [
        load_timeline(label, path, role)
        for role, entries in (("final", args.timeline), ("before", args.before_timeline))
        for label, path in entries
    ]
    sources = {op["source_sha256"] for op in operations if op["role"] == "final"}
    sources.update(trace["source_sha256"] for trace in traces if trace["role"] == "final")
    require(len(sources) <= 1, "Final NCU/NSYS evidence comes from different package sources")
    comparisons = compare_captures(operations, profiles)
    timeline_comparisons = compare_timelines(traces)
    args.output.mkdir(parents=True, exist_ok=False)
    result = {
        "schema_version": 1,
        "analysis_created_at_utc": datetime.now(UTC).isoformat(),
        "analysis_script_sha256": sha256(Path(__file__)),
        "decoder_script_sha256": sha256(Path(decoder.__file__)),
        "measurement_policy": (
            "Analysis of saved captures. Nsight Compute reports replay kernel time. "
            "Nsight Systems reports instrumented application intervals."
        ),
        "missing_counter_policy": (
            "JSON null / blank CSV / unavailable Markdown. "
            "operation sums null if any constituent is missing"
        ),
        "flops_definition": (
            "predicated-thread ADD + MUL + 2*FMA. Conversions, atomics, comparisons, "
            "integer work and divide/sqrt helpers have separate counters."
        ),
        "roofline_definition": (
            "Official L1/L2 outgoing writeback proxy bytes. Total requested-sector bytes "
            "include atomic and reduction requests and have their own intensity columns. "
            "Ceilings use the NVIDIA theoretical peak-sustained model at each recorded clock."
        ),
        "operations": operations,
        "profiles": profiles,
        "timelines": traces,
        "timeline_comparisons": timeline_comparisons,
        "comparisons": comparisons,
        "figures": {},
    }
    if not args.no_plots and profiles:
        require(
            args.font and args.font.is_file(),
            "Pass an existing --font Geist-Regular.ttf or --no-plots",
        )
        decoder.font_manager.fontManager.addfont(args.font)
        decoder.plt.rcParams.update(
            {
                "font.family": "Geist",
                "font.size": 10,
                "text.color": "black",
                "axes.labelcolor": "black",
                "figure.facecolor": "white",
                "axes.facecolor": "white",
                "pdf.fonttype": 42,
                "svg.fonttype": "none",
            }
        )
        for precision in ("fp32", "fp64"):
            result["figures"][precision] = roofline(profiles, operations, precision, args.output)
    write_csv(args.output / "counters.csv", counters_rows(profiles))
    write_csv(args.output / "before_after.csv", comparisons)
    write_csv(
        args.output / "timeline.csv",
        [
            {
                "capture": t["id"],
                "role": t["role"],
                "precision": t["precision"],
                "requested_gradients": "+".join(t["requested_gradients"])
                if t["requested_gradients"] is not None
                else None,
                "pose_requires_grad": t["pose_requires_grad"],
                "source_sha256": t["source_sha256"],
                "views": t["views"],
                **row,
            }
            for t in traces
            for row in t["ranges"]
        ],
    )
    (args.output / "profile_summary.md").write_text(markdown(result))
    (args.output / "profile_report.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "NCU_captures": len(operations),
                "kernels": len(profiles),
                "NSYS_captures": len(traces),
                "figures": result["figures"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
