#!/usr/bin/env bash
# Profile the two complete material VJPs using a finished scalar-profile metric selection.
set -euo pipefail

DPT_WORK=${DPT_WORK:?Set DPT_WORK to the external benchmark work directory}
DPT_CT=${DPT_CT:?Set DPT_CT to the downloaded native CTChest.nii.gz}
DPT_SOURCE=${DPT_SOURCE:-"$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"}
DPT_REPORT=${DPT_REPORT:-"$DPT_WORK/profiles-material"}
DPT_PROFILE_SETUP=${DPT_PROFILE_SETUP:-"$DPT_WORK/profiles"}
DPT_BASELINE=${DPT_BASELINE:-"$DPT_SOURCE/docs/assets/results/baseline-chest-512.json"}
DPT_PYTHON=${DPT_PYTHON:-"$(command -v python3)"}
DPT_NCU=${DPT_NCU:-"$(command -v ncu)"}
DPT_NSYS=${DPT_NSYS:-"$(command -v nsys)"}
DPT_PROFILE_DRIVER=${DPT_PROFILE_DRIVER:-"$DPT_SOURCE/benchmarks/profile.py"}
DPT_PROFILE_CACHE=${DPT_PROFILE_CACHE:-"$DPT_WORK/cache/profile"}

export PYTHONPATH="$DPT_SOURCE/python"
# The reused benchmark records both profiler versions during preparation.
# sudo may supply a restricted PATH even when NCU itself is invoked absolutely.
export PATH="$(dirname "$DPT_NCU"):$(dirname "$DPT_NSYS"):$PATH"
export WARP_CACHE_PATH="$DPT_PROFILE_CACHE/warp"
export CUDA_CACHE_PATH="$DPT_PROFILE_CACHE/cuda"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
if [[ -e "$DPT_REPORT" ]]; then
  echo "Refusing to replace material profile output: $DPT_REPORT" >&2
  exit 1
fi
test -f "$DPT_PROFILE_SETUP/completed-at.txt"
test -f "$DPT_PROFILE_SETUP/metrics-selected.txt"
test -f "$DPT_PROFILE_SETUP/sections-selected.txt"
test -x "$DPT_NCU"
test -x "$DPT_NSYS"
mkdir -p "$DPT_WORK"
exec 9>"$DPT_WORK/gpu.lock"
flock 9
mkdir -p "$DPT_REPORT" "$WARP_CACHE_PATH" "$CUDA_CACHE_PATH"
date -u --iso-8601=seconds > "$DPT_REPORT/started-at.txt"
finish() {
  local code=$?
  echo "$code" > "$DPT_REPORT/exit-code.txt"
  date -u --iso-8601=seconds > "$DPT_REPORT/stopped-at.txt"
  if [[ $(id -u) == 0 ]]; then
    local owner
    owner=$(stat -c '%u:%g' "$DPT_SOURCE")
    chown -R "$owner" "$DPT_REPORT" "$DPT_PROFILE_CACHE"
  fi
}
trap finish EXIT
for file in ncu-version.txt sections-selected.txt metrics-selected.txt metric-availability.json; do
  cp "$DPT_PROFILE_SETUP/$file" "$DPT_REPORT/$file"
done
cp -r "$DPT_PROFILE_SETUP/sections" "$DPT_REPORT/sections"
"$DPT_NCU" --version > "$DPT_REPORT/ncu-version-current.txt"
cmp "$DPT_REPORT/ncu-version.txt" "$DPT_REPORT/ncu-version-current.txt"
printf '%s\n' "$DPT_PROFILE_SETUP" > "$DPT_REPORT/metric-selection-from.txt"
nvidia-smi -q > "$DPT_REPORT/nvidia-smi-before.txt"
mapfile -t DPT_SECTIONS < "$DPT_REPORT/sections-selected.txt"
DPT_SECTION_ARGS=()
for section in "${DPT_SECTIONS[@]}"; do
  DPT_SECTION_ARGS+=(--section "$section")
done
DPT_METRICS=$(cat "$DPT_REPORT/metrics-selected.txt")
for mode in fp32 fp64; do
  if [[ $(id -u) == 0 ]]; then
    chown -R "$(stat -c '%u:%g' "$DPT_SOURCE")" "$DPT_REPORT" "$DPT_PROFILE_CACHE"
  fi
  echo "NCU $mode complete material VJP -> $DPT_REPORT"
  "$DPT_NCU" --profile-from-start off --nvtx --nvtx-include operation/ \
    --range-filter yes:1:1 --replay-mode kernel --cache-control all --clock-control none \
    --import-source yes "${DPT_SECTION_ARGS[@]}" --metrics "$DPT_METRICS" \
    -o "$DPT_REPORT/$mode-material-vjp" \
    "$DPT_PYTHON" "$DPT_PROFILE_DRIVER" --source "$DPT_SOURCE" \
    --baseline "$DPT_BASELINE" --ct "$DPT_CT" --mode "$mode" --operation material-vjp --views ap \
    --metadata "$DPT_REPORT/$mode-material-vjp.metadata.json"
  DPT_CAPTURE="$DPT_REPORT/$mode-material-vjp.ncu-repz"
  if [[ ! -f "$DPT_CAPTURE" ]]; then
    DPT_CAPTURE="$DPT_REPORT/$mode-material-vjp.ncu-rep"
  fi
  test -f "$DPT_CAPTURE"
  "$DPT_NCU" --import "$DPT_CAPTURE" --csv --page raw --print-units base \
    > "$DPT_REPORT/$mode-material-vjp.csv"
  "$DPT_NCU" --import "$DPT_CAPTURE" --csv --page raw --print-units base \
    --metrics sass__thread_inst_executed_true_per_opcode,sass__thread_inst_executed_per_opcode_category \
    --print-metric-instances details > "$DPT_REPORT/$mode-material-vjp-instances.csv"
  "$DPT_NCU" --import "$DPT_CAPTURE" --csv --page source --print-source cuda,sass \
    --metrics smsp__pcsamp_warps_issue_stalled_short_scoreboard,smsp__pcsamp_warps_issue_stalled_wait \
    > "$DPT_REPORT/$mode-material-vjp-source.csv"
done
nvidia-smi -q > "$DPT_REPORT/nvidia-smi-after.txt"
date -u --iso-8601=seconds > "$DPT_REPORT/completed-at.txt"
echo "Completed both material VJP captures: $DPT_REPORT"
