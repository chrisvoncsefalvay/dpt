"""Check exact dynamic SASS counts from saved Nsight Compute report imports.

Pass a generated fullscan profile report and one or more LABEL=DIRECTORY
imports. Each directory contains instructions.csv and its receipt.json. The
output is compact JSON. This command does not import DPT or execute CUDA.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def timestamp(value):
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    require(result.utcoffset() is not None, "A recorded timestamp has no timezone")
    return result


def read_instructions(path):
    """Deduplicate inline aliases by kernel name and instruction address."""
    kernels, kernel, header = {}, None, None
    with path.open(newline="") as stream:
        for row in csv.reader(stream):
            if not row:
                continue
            if row[0] in ("Kernel Name", "Function Name"):
                kernel = row[1]
                continue
            if "Address" in row and "Source" in row:
                header = row
                require(
                    "Instructions Executed" in header
                    and "Predicated-On Thread Instructions Executed" in header,
                    "The import lacks exact dynamic instruction counts",
                )
                continue
            if kernel is None or header is None:
                continue
            indices = {
                name: header.index(name)
                for name in (
                    "Address",
                    "Source",
                    "Instructions Executed",
                    "Predicated-On Thread Instructions Executed",
                )
            }
            if len(row) <= max(indices.values()):
                continue
            address = row[indices["Address"]]
            if not re.fullmatch(r"0x[0-9a-fA-F]+", address):
                continue
            words = row[indices["Source"]].strip().split()
            if words and words[0].startswith("@"):
                words.pop(0)
            require(bool(words), "Instruction address has no disassembly")
            counts = []
            for name in ("Instructions Executed", "Predicated-On Thread Instructions Executed"):
                value = row[indices[name]].replace(",", "")
                require(value.isdigit(), "Exact instruction count is missing or non-integer")
                counts.append(int(value))
            instruction = (" ".join(words), *counts)
            previous = kernels.setdefault(kernel, {}).setdefault(address, instruction)
            require(previous == instruction, "Conflicting duplicate instruction address")
    require(bool(kernels), "No dynamic instructions were imported")
    return kernels


def labelled(value):
    label, separator, location = value.partition("=")
    require(bool(separator and label and location), "Expected LABEL=PATH")
    return label, Path(location)


def checked_import(label, directory, profile_report):
    operations = [row for row in profile_report["operations"] if row["id"] == label]
    require(len(operations) == 1, "Import label must identify one profiled operation")
    operation = operations[0]
    require(operation["file_integrity"]["status"] == "verified_exports", "Unbound capture")
    receipt_path, csv_path = directory / "receipt.json", directory / "instructions.csv"
    receipt = json.loads(receipt_path.read_text())
    require(
        receipt["returncode"] == 0
        and receipt["report_unchanged"] is True
        and receipt["capture_metadata_sha256"] == operation["capture_metadata"]["sha256"]
        and receipt["source_sha256"] == operation["source_sha256"]
        and receipt["report"] in operation["reports"]
        and receipt["export_sha256"] == sha256(csv_path),
        "Import binary, source, capture or CSV hash differs",
    )
    require(
        timestamp(operation["measurement_completed_at"])
        <= timestamp(receipt["started_at_utc"])
        <= timestamp(receipt["completed_at_utc"]),
        "Import dates precede the completed capture or are reversed",
    )
    command = receipt["command"]
    for flag, value in (("--page", "source"), ("--print-source", "sass")):
        require(
            flag in command and command[command.index(flag) + 1] == value,
            "Receipt is not the expected source import",
        )
    require(
        "--import" in command
        and Path(command[command.index("--import") + 1]).name == receipt["report"]["file"],
        "Receipt command imported a different report",
    )
    imported = read_instructions(csv_path)
    require(len(imported) == 1, "Each import must contain exactly one hot kernel")
    kernel_name, instructions = next(iter(imported.items()))
    matches = [
        row
        for row in profile_report["profiles"]
        if row["operation_id"] == label and row["kernel"] == kernel_name
    ]
    require(len(matches) == 1, "Imported kernel does not identify one recorded launch")
    kernel = matches[0]
    static, warp, predicated, basic = Counter(), Counter(), Counter(), Counter()
    for instruction, warp_count, thread_count in instructions.values():
        opcode = instruction.split()[0]
        static[opcode] += 1
        warp[opcode] += warp_count
        predicated[opcode] += thread_count
        basic[opcode.split(".")[0]] += thread_count
    recorded = kernel["predicated_thread_opcode_counts"]
    require(bool(recorded), "Recorded dynamic opcode totals are unavailable")
    require(
        {key: value for key, value in basic.items() if value}
        == {key: value for key, value in recorded.items() if value},
        "Per-address instruction counts differ from recorded opcode totals",
    )
    flops = {}
    for precision, prefix in (("fp32", "F"), ("fp64", "D")):
        value = basic[prefix + "ADD"] + basic[prefix + "MUL"] + 2 * basic[prefix + "FMA"]
        require(
            value == kernel["precision"][precision]["add_mul_fma_flops"],
            "Per-address FLOPs differ from the recorded profile",
        )
        flops[precision] = value
    return {
        "operation_id": label,
        "source_sha256": operation["source_sha256"],
        "capture_metadata_sha256": operation["capture_metadata"]["sha256"],
        "capture_completed_at_utc": operation["measurement_completed_at"],
        "workload": operation["workload"],
        "binary_report": receipt["report"],
        "receipt_sha256": sha256(receipt_path),
        "instruction_csv_sha256": sha256(csv_path),
        "import_started_at_utc": receipt["started_at_utc"],
        "import_completed_at_utc": receipt["completed_at_utc"],
        "import_command": command,
        "kernel_id": kernel["ncu_kernel_id"],
        "kernel_name": kernel_name,
        "unique_instruction_addresses": len(instructions),
        "static_instruction_sites_by_opcode_modifier": dict(sorted(static.items())),
        "warp_instructions_by_opcode_modifier": dict(sorted(warp.items())),
        "predicated_thread_instructions_by_opcode_modifier": dict(sorted(predicated.items())),
        "predicated_thread_instructions_by_basic_opcode": dict(sorted(basic.items())),
        "recorded_predicated_thread_category_counts": kernel["predicated_thread_category_counts"],
        "add_mul_2fma_flops": flops,
        "reciprocal_thread_instructions": {
            "MUFU.RCP": predicated["MUFU.RCP"],
            "MUFU.RCP64H": predicated["MUFU.RCP64H"],
        },
        "all_dynamic_opcode_totals_match": True,
        "all_flop_totals_match": True,
    }


def collect(profile_path, imports, comparisons):
    profile_report = json.loads(profile_path.read_text())
    labels = [label for label, _ in imports]
    require(len(labels) == len(set(labels)), "Duplicate import label")
    rows = [checked_import(label, directory, profile_report) for label, directory in imports]
    by_label = {row["operation_id"]: row for row in rows}
    operations = {row["id"]: row for row in profile_report["operations"]}
    pairs = []
    for before, after in comparisons:
        require(before in by_label and after in by_label, "Comparison is missing an import")
        signature = operations[before]["comparison_signature"]
        require(
            signature is not None and signature == operations[after]["comparison_signature"],
            "Compared captures do not have identical workload and profiler settings",
        )
        left, right = by_label[before], by_label[after]
        differences = {}
        quantities = {
            "MUFU.RCP64H": lambda row: row["reciprocal_thread_instructions"]["MUFU.RCP64H"],
            "FP64_ADD_MUL_2FMA_FLOPs": lambda row: row["add_mul_2fma_flops"]["fp64"],
            "conversion_thread_instructions": lambda row: row[
                "recorded_predicated_thread_category_counts"
            ].get("Conversion"),
            "REDG_thread_instructions": lambda row: row[
                "predicated_thread_instructions_by_basic_opcode"
            ].get("REDG", 0),
        }
        for name, getter in quantities.items():
            old, new = getter(left), getter(right)
            differences[name] = {
                "before": old,
                "after": new,
                "difference": new - old if old is not None and new is not None else None,
                "percent_change": 100 * (new / old - 1)
                if old not in (None, 0) and new is not None
                else None,
            }
        pairs.append({"before": before, "after": after, "quantities": differences})
    return {
        "schema_version": 1,
        "analysis_created_at_utc": datetime.now(UTC).isoformat(),
        "analysis_script_sha256": sha256(Path(__file__)),
        "profile_report_sha256": sha256(profile_path),
        "profile_analysis_created_at_utc": profile_report["analysis_created_at_utc"],
        "count_definition": "Exact instruction-address counts from read-only saved-report imports",
        "flops_definition": "Predicated thread ADD + MUL + 2 x FMA",
        "timing_boundary": "Profiler diagnostics. No clean benchmark timings are inferred",
        "imports": rows,
        "comparisons": pairs,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-report", type=Path, required=True)
    parser.add_argument("--import", dest="imports", action="append", required=True)
    parser.add_argument("--compare", action="append", default=[], help="BEFORE_LABEL=AFTER_LABEL")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "Preserve an existing output file")
    imports = [labelled(value) for value in args.imports]
    pairs = []
    for value in args.compare:
        before, separator, after = value.partition("=")
        require(bool(separator and before and after), "Expected BEFORE_LABEL=AFTER_LABEL")
        pairs.append((before, after))
    result = collect(args.profile_report, imports, pairs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output), "verified_imports": len(result["imports"])}))


if __name__ == "__main__":
    main()
