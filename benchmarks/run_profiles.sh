#!/usr/bin/env bash
# Profile completed operations on a GPU with access to performance counters.
# Detach externally with nohup/setsid. A shared flock serialises GPU work.
set -euo pipefail

DPT_WORK=${DPT_WORK:?Set DPT_WORK to an external profiling output/cache directory}
DPT_CT=${DPT_CT:?Set DPT_CT to the downloaded native CTChest.nii.gz}
DPT_SOURCE=${DPT_SOURCE:-"$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"}
DPT_REPORT=${DPT_REPORT:-"$DPT_WORK/profiles"}
DPT_BASELINE=${DPT_BASELINE:-"$DPT_SOURCE/docs/assets/results/baseline-chest-512.json"}
DPT_PYTHON=${DPT_PYTHON:-"$(command -v python3)"}
DPT_NCU=${DPT_NCU:-"$(command -v ncu)"}
DPT_NSYS=${DPT_NSYS:-"$(command -v nsys)"}
DPT_PROFILE_DRIVER=${DPT_PROFILE_DRIVER:-"$DPT_SOURCE/benchmarks/profile.py"}
DPT_PROFILE_CACHE=${DPT_PROFILE_CACHE:-"$DPT_WORK/cache/profile"}

export PYTHONPATH="$DPT_SOURCE/python"
export WARP_CACHE_PATH="$DPT_PROFILE_CACHE/warp"
export CUDA_CACHE_PATH="$DPT_PROFILE_CACHE/cuda"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2

mkdir -p "$DPT_REPORT" "$WARP_CACHE_PATH" "$CUDA_CACHE_PATH"
if [[ -e "$DPT_REPORT/started-at.txt" ]]; then
  echo "Refusing to overwrite an existing profiling run: $DPT_REPORT" >&2
  exit 1
fi
exec 9>"$DPT_WORK/gpu.lock"
flock 9
date -u --iso-8601=seconds > "$DPT_REPORT/started-at.txt"

prepare_writable_paths() {
  if [[ $(id -u) == 0 ]]; then
    local owner
    owner=$(stat -c '%u:%g' "$DPT_SOURCE")
    chown -R "$owner" "$DPT_REPORT" "$DPT_PROFILE_CACHE"
  fi
}

finish() {
  local result=$?
  echo "$result" > "$DPT_REPORT/exit-code.txt"
  date -u --iso-8601=seconds > "$DPT_REPORT/stopped-at.txt"
  prepare_writable_paths
}
trap finish EXIT
prepare_writable_paths

"$DPT_NSYS" --version > "$DPT_REPORT/nsys-version.txt"
"$DPT_NCU" --version > "$DPT_REPORT/ncu-version.txt"
"$DPT_NCU" --list-sections > "$DPT_REPORT/ncu-sections.txt"
# Query only metadata. This requires counter permissions in the GPU driver.
"$DPT_NCU" --query-metrics --query-metrics-mode all --devices 0 \
  > "$DPT_REPORT/ncu-metrics-profiling.txt"
"$DPT_NCU" --query-metrics --query-metrics-mode all --query-metrics-collection source --devices 0 \
  > "$DPT_REPORT/ncu-metrics-source.txt"
nvidia-smi -q > "$DPT_REPORT/nvidia-smi-before.txt"

"$DPT_PYTHON" - "$DPT_REPORT" "$DPT_NCU" <<'PY'
import json
import re
import shutil
import sys
from pathlib import Path
report, ncu = Path(sys.argv[1]), Path(sys.argv[2])
available_sections = (report / 'ncu-sections.txt').read_text()
desired_sections = ['SpeedOfLight', 'LaunchStats', 'Occupancy', 'MemoryWorkloadAnalysis',
                    'ComputeWorkloadAnalysis', 'InstructionStats', 'SchedulerStats',
                    'SourceCounters', 'WarpStateStats',
                    'SpeedOfLight_HierarchicalSingleRooflineChart',
                    'SpeedOfLight_HierarchicalDoubleRooflineChart']
sections = [s for s in desired_sections if re.search(r'(?<!\w)' + re.escape(s) + r'(?!\w)', available_sections)]
if not {'SpeedOfLight', 'LaunchStats', 'SourceCounters'}.issubset(sections):
    raise SystemExit('Required NCU sections absent; inspect ncu-sections.txt')
query = '\n'.join((report / name).read_text() for name in ('ncu-metrics-profiling.txt', 'ncu-metrics-source.txt'))
available = set(re.findall(r'\b[a-zA-Z][a-zA-Z0-9_]*__[a-zA-Z0-9_.]+', query))
desired = [
    'dram__bytes.sum', 'dram__bytes_read.sum', 'dram__bytes_write.sum',
    'l1tex__t_bytes.sum', 'l1tex__t_sectors.sum',
    'l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum',
    'l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum',
    'l1tex__t_sectors_pipe_lsu_mem_global_op_atom.sum',
    'l1tex__t_sectors_pipe_lsu_mem_global_op_red.sum',
    'l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum',
    'l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum',
    'l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum',
    'l1tex__t_requests_pipe_lsu_mem_global_op_st.sum',
    'l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum',
    'l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum',
    'l1tex__lsu_writeback_active_mem_lgds.sum',
    'l1tex__lsu_writeback_active_mem_lg.sum',
    'lts__t_bytes.sum', 'lts__t_sectors.sum',
    'lts__t_sectors_op_read.sum', 'lts__t_sectors_op_write.sum',
    'lts__t_sectors_op_atom.sum', 'lts__t_sectors_op_red.sum',
    'lts__t_sectors_lookup_hit.sum', 'lts__t_sectors_lookup_miss.sum',
    'lts__lts2xbar_cycles_active.sum',
    'sass__thread_inst_executed_true_per_opcode',
    'sass__thread_inst_executed_per_opcode_category',
    'sass__inst_executed_register_spilling',
]
desired += [f'smsp__sass_thread_inst_executed_op_{p}{op}_pred_on.sum'
            for p in ('f', 'd') for op in ('add', 'mul', 'fma')]
selected = [m for m in desired if m in available]
(report / 'sections-selected.txt').write_text('\n'.join(sections) + '\n')
(report / 'metrics-selected.txt').write_text(','.join(selected) + '\n')
(report / 'metric-availability.json').write_text(json.dumps({
    'selected_sections': sections, 'unavailable_sections': [s for s in desired_sections if s not in sections],
    'selected_metrics': selected, 'unavailable_metrics': [m for m in desired if m not in selected],
    'absence_policy': 'Unavailable counters remain null; no estimate or zero substitution.'}, indent=2) + '\n')
(report / 'sections').mkdir(exist_ok=True)
for source in ncu.parent.rglob('*.section'):
    if source.stem in sections:
        shutil.copy2(source, report / 'sections' / source.name)
PY

mapfile -t DPT_SECTIONS < "$DPT_REPORT/sections-selected.txt"
DPT_SECTION_ARGS=()
for section in "${DPT_SECTIONS[@]}"; do
  DPT_SECTION_ARGS+=(--section "$section")
done
DPT_METRICS=$(cat "$DPT_REPORT/metrics-selected.txt")
DPT_METRIC_ARGS=()
if [[ -n "$DPT_METRICS" ]]; then
  DPT_METRIC_ARGS=(--metrics "$DPT_METRICS")
fi

for mode in fp32 fp64; do
  for operation in public iteration; do
    echo "NSYS $mode $operation -> $DPT_REPORT"
    prepare_writable_paths
    "$DPT_NSYS" profile --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none \
      --capture-range=cudaProfilerApi --capture-range-end=stop \
      -o "$DPT_REPORT/$mode-$operation" \
      "$DPT_PYTHON" "$DPT_PROFILE_DRIVER" --source "$DPT_SOURCE" \
      --baseline "$DPT_BASELINE" --ct "$DPT_CT" --mode "$mode" --operation "$operation" \
      --views ap lateral oblique --metadata "$DPT_REPORT/$mode-$operation.metadata.json"
    "$DPT_NSYS" stats --report cuda_gpu_kern_sum,cuda_api_sum,nvtx_sum \
      --format csv --output "$DPT_REPORT/$mode-$operation-stats" \
      "$DPT_REPORT/$mode-$operation.nsys-rep"
  done
  for operation in forward volume pose; do
    echo "NCU complete $mode $operation -> $DPT_REPORT"
    prepare_writable_paths
    "$DPT_NCU" --profile-from-start off --nvtx --nvtx-include operation/ \
      --range-filter yes:1:1 --replay-mode kernel --cache-control all --clock-control none \
      --import-source yes "${DPT_SECTION_ARGS[@]}" "${DPT_METRIC_ARGS[@]}" \
      -o "$DPT_REPORT/$mode-$operation" \
      "$DPT_PYTHON" "$DPT_PROFILE_DRIVER" --source "$DPT_SOURCE" \
      --baseline "$DPT_BASELINE" --ct "$DPT_CT" --mode "$mode" --operation "$operation" \
      --views ap --metadata "$DPT_REPORT/$mode-$operation.metadata.json"
    # NCU 2026.3.1 writes compressed .ncu-repz reports on this installation.
    DPT_CAPTURE="$DPT_REPORT/$mode-$operation.ncu-repz"
    if [[ ! -f "$DPT_CAPTURE" ]]; then
      DPT_CAPTURE="$DPT_REPORT/$mode-$operation.ncu-rep"
    fi
    [[ -f "$DPT_CAPTURE" ]]
    "$DPT_NCU" --import "$DPT_CAPTURE" --csv --page raw --print-units base \
      > "$DPT_REPORT/$mode-$operation.csv"
    "$DPT_NCU" --import "$DPT_CAPTURE" --csv --page raw --print-units base \
      --metrics sass__thread_inst_executed_true_per_opcode,sass__thread_inst_executed_per_opcode_category \
      --print-metric-instances details > "$DPT_REPORT/$mode-$operation-instances.csv"
    "$DPT_NCU" --import "$DPT_CAPTURE" --csv --page source --print-source cuda,sass \
      --metrics smsp__pcsamp_warps_issue_stalled_short_scoreboard,smsp__pcsamp_warps_issue_stalled_wait \
      > "$DPT_REPORT/$mode-$operation-source.csv"
  done
done
nvidia-smi -q > "$DPT_REPORT/nvidia-smi-after.txt"
date -u --iso-8601=seconds > "$DPT_REPORT/completed-at.txt"
echo "Completed profiler reports: $DPT_REPORT"
