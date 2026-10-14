#!/usr/bin/env bash
# Interactive launcher for DeepLearnLib binaries (Docker and local Linux).
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${ROOT}/build}"
DEFAULT_EXPERIMENTS_JSON="${ROOT}/config/experiments.json"
export EXPERIMENTS_JSON="${EXPERIMENTS_JSON:-${DEFAULT_EXPERIMENTS_JSON}}"
# Full-run progress (options 16 / 17). Empty means do not record or skip steps.
# Completed jobs stay in the JSON until that file is deleted by hand.
MENU_RESUME_RUN=""
MENU_RESUME_HAD_FAILURE=0
# shellcheck source=cuda_env.sh
source "${ROOT}/scripts/cuda_env.sh"

progress_json_file() {
  printf '%s\n' "${ROOT}/results/run_all_progress.json"
}

resume_step_id() {
  local IFS=' '
  printf '%s' "$*"
}

# Args: <has|mark|list> <json-path> [job-id] [run-name]
progress_py() {
  local py
  if ! py="$(python_bin)"; then
    echo "[menu] Python not found; cannot read or update the progress file." >&2
    return 1
  fi
  "${py}" - "$@" <<'PY'
import json
import os
import sys
from datetime import datetime, timezone

cmd = sys.argv[1]
path = sys.argv[2]


def fail(message):
    print(f"[menu] {message}", file=sys.stderr)
    raise SystemExit(2)


def load():
    if not os.path.isfile(path):
        return {"jobs": {}}
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except json.JSONDecodeError as exc:
        fail(f"Cannot parse {path}: {exc}. Fix the file or delete it to reset.")
    if not isinstance(data, dict) or not isinstance(data.get("jobs"), dict):
        fail(f"Invalid progress file {path} (expected a 'jobs' object). Fix it or delete it to reset.")
    return data


def is_done(entry):
    return isinstance(entry, dict) and entry.get("status") == "done"


def save(data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(data, handle, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


if cmd == "has":
    job = sys.argv[3]
    entry = load()["jobs"].get(job)
    raise SystemExit(0 if is_done(entry) else 1)

if cmd == "mark":
    job = sys.argv[3]
    run_name = sys.argv[4] if len(sys.argv) > 4 else ""
    data = load()
    jobs = data["jobs"]
    if not is_done(jobs.get(job)):
        jobs[job] = {
            "status": "done",
            "finished_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "run": run_name,
        }
        save(data)
    raise SystemExit(0)

if cmd == "list":
    for job, entry in load()["jobs"].items():
        if is_done(entry):
            print(job)
    raise SystemExit(0)

fail(f"Unknown progress command '{cmd}'.")
PY
}

resume_is_done() {
  local id="$1"
  local status
  [[ -n "${MENU_RESUME_RUN}" ]] || return 1
  progress_py has "$(progress_json_file)" "${id}"
  status=$?
  if [[ "${status}" -eq 0 ]]; then
    return 0
  fi
  if [[ "${status}" -eq 1 ]]; then
    return 1
  fi
  echo "[menu] Aborting because the progress file could not be read." >&2
  exit "${status}"
}

resume_mark_done() {
  local id="$1"
  [[ -n "${MENU_RESUME_RUN}" ]] || return 0
  mkdir -p "${ROOT}/results"
  progress_py mark "$(progress_json_file)" "${id}" "${MENU_RESUME_RUN}"
}

import_legacy_resume_files() {
  local legacy id
  local -a legacy_files=(
    "${ROOT}/results/.menu_run_all.resume"
    "${ROOT}/results/.menu_run_all_training.resume"
  )
  for legacy in "${legacy_files[@]}"; do
    [[ -f "${legacy}" ]] || continue
    echo "[menu] Importing legacy progress from ${legacy}"
    while IFS= read -r id || [[ -n "${id}" ]]; do
      id="${id//$'\r'/}"
      [[ -n "${id}" ]] || continue
      resume_mark_done "${id}" || return 1
    done < "${legacy}"
    mv "${legacy}" "${legacy}.imported"
  done
}

begin_full_run() {
  MENU_RESUME_RUN="$1"
  MENU_RESUME_HAD_FAILURE=0
  mkdir -p "${ROOT}/results"
  import_legacy_resume_files || exit 1
  local file done_jobs
  file="$(progress_json_file)"
  echo "[menu] Starting ${MENU_RESUME_RUN}"
  echo "[menu] Progress file: ${file}"
  echo "[menu] Completed jobs are skipped. Delete this file, or remove one job entry, to run them again."
  if ! done_jobs="$(progress_py list "${file}")"; then
    echo "[menu] Aborting because the progress file could not be read." >&2
    exit 1
  fi
  if [[ -n "${done_jobs}" ]]; then
    echo "[menu] Already completed:"
    printf '%s\n' "${done_jobs}" | sed 's/^/[menu]   /'
  else
    echo "[menu] No completed jobs recorded yet."
  fi
}

finish_full_run() {
  local file
  file="$(progress_json_file)"
  if [[ "${MENU_RESUME_HAD_FAILURE}" -eq 0 ]]; then
    echo "[menu] ${MENU_RESUME_RUN} completed."
  else
    echo "[menu] ${MENU_RESUME_RUN} finished with failures. Failed jobs were not recorded and will run again."
  fi
  echo "[menu] Progress kept in ${file}. Reset is manual: delete that file to run completed jobs again."
  MENU_RESUME_RUN=""
}

cached_cuda_arch() {
  local cache="${BUILD_DIR}/CMakeCache.txt"
  if [[ ! -f "${cache}" ]]; then
    return 1
  fi
  grep -E '^CMAKE_CUDA_ARCHITECTURES(:[^=]*)?=' "${cache}" | head -n 1 | sed 's/^[^=]*=//'
}

ensure_built() {
  local target="$1"
  if [[ ! -f "${BUILD_DIR}/CMakeCache.txt" ]]; then
    echo "[menu] No build tree at ${BUILD_DIR}. Run ./scripts/dev.sh first." >&2
    return 1
  fi
  if [[ -n "${CMAKE_CUDA_ARCHITECTURES:-}" ]]; then
    local cached
    cached="$(cached_cuda_arch || true)"
    if [[ -n "${cached}" && "${cached}" != "${CMAKE_CUDA_ARCHITECTURES}" ]]; then
      echo "[menu] CUDA arch '${cached}' != GPU '${CMAKE_CUDA_ARCHITECTURES}'; reconfiguring"
      local cmake_args=(-S "${ROOT}" -B "${BUILD_DIR}" -DCMAKE_CUDA_ARCHITECTURES="${CMAKE_CUDA_ARCHITECTURES}")
      if [[ -n "${TORCH_CUDA_ARCH_LIST:-}" ]]; then
        cmake_args+=(-DTORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}")
      fi
      cmake "${cmake_args[@]}"
    fi
  fi
  echo "[menu] Building ${target}"
  if command -v ninja >/dev/null 2>&1 && [[ -f "${BUILD_DIR}/build.ninja" ]]; then
    ninja -C "${BUILD_DIR}" "${target}"
  else
    cmake --build "${BUILD_DIR}" --parallel --target "${target}"
  fi
}

find_bin() {
  local name="$1"
  local candidates=(
    "${BUILD_DIR}/tests/${name}"
    "${BUILD_DIR}/tests/${name}.exe"
    "${BUILD_DIR}/benchmarks/${name}"
    "${BUILD_DIR}/benchmarks/${name}.exe"
    "${BUILD_DIR}/${name}"
    "${BUILD_DIR}/${name}.exe"
  )
  local path
  for path in "${candidates[@]}"; do
    if [[ -x "${path}" ]]; then
      printf '%s\n' "${path}"
      return 0
    fi
  done
  return 1
}

run_bin() {
  local name="$1"
  shift || true
  if ! ensure_built "${name}"; then
    return 1
  fi
  local bin
  if ! bin="$(find_bin "${name}")"; then
    echo "[menu] Binary '${name}' not found under ${BUILD_DIR}." >&2
    echo "[menu] Build first (./scripts/dev.sh or cmake --build ${BUILD_DIR})." >&2
    return 1
  fi
  echo "[menu] Running ${bin} $*"
  (cd "$(dirname "${bin}")" && "${bin}" "$@")
}

is_torch_target() {
  local name="$1"
  [[ "${name}" == *_torch || "${name}" == bench_micro_ops ]]
}

# Used by Run All / Sanity: try to compile the target, then run it.
# Skip (do not abort the sequence) if the target is not in this build
# (typical for Torch binaries when LibTorch was not found) or if it fails.
run_optional() {
  local name="$1"
  shift || true
  local id
  id="$(resume_step_id "${name}" "$@")"
  if resume_is_done "${id}"; then
    echo "[menu] Resume: skipping already completed '${id}'"
    return 0
  fi
  if ! ensure_built "${name}"; then
    if is_torch_target "${name}"; then
      echo "[menu] Skipping '${name}' (not in this build — Torch baselines need LibTorch)."
    else
      echo "[menu] Skipping '${name}' (compile failed). Custom targets should exist after a successful CMake configure with OpenCV."
    fi
    return 0
  fi
  local bin
  if ! bin="$(find_bin "${name}")"; then
    echo "[menu] Skipping '${name}' (ninja succeeded but the executable is not under ${BUILD_DIR})."
    return 0
  fi
  echo "[menu] Running ${bin} $*"
  if ! (cd "$(dirname "${bin}")" && "${bin}" "$@"); then
    echo "[menu] WARNING: '${name}' failed; continuing."
    MENU_RESUME_HAD_FAILURE=1
    return 0
  fi
  resume_mark_done "${id}"
}

run_plots_optional() {
  local id="plot_metrics"
  if [[ -n "${MENU_RESUME_RUN}" ]]; then
    id="plot_metrics ${MENU_RESUME_RUN}"
  fi
  if resume_is_done "${id}"; then
    echo "[menu] Resume: skipping already completed '${id}'"
    return 0
  fi
  if run_plots; then
    resume_mark_done "${id}"
  else
    echo "[menu] WARNING: plotting failed; continuing."
    MENU_RESUME_HAD_FAILURE=1
  fi
}

python_bin() {
  if command -v python3 >/dev/null 2>&1; then
    command -v python3
    return 0
  fi
  if command -v python >/dev/null 2>&1; then
    command -v python
    return 0
  fi
  return 1
}

run_setup_datasets() {
  local py
  if ! py="$(python_bin)"; then
    echo "[menu] Python not found; cannot run setup_datasets.py." >&2
    return 1
  fi
  echo "[menu] Running ${ROOT}/scripts/setup_datasets.py"
  "${py}" "${ROOT}/scripts/setup_datasets.py" --data-root "${ROOT}/data"
}

run_plots() {
  local py
  if ! py="$(python_bin)"; then
    echo "[menu] Python not found; cannot run plot_metrics.py." >&2
    return 1
  fi
  echo "[menu] Running ${ROOT}/scripts/plot_metrics.py"
  "${py}" "${ROOT}/scripts/plot_metrics.py" --results-root "${ROOT}/results"
}

run_all_training() {
  echo
  echo "[menu] === Run all full training pipelines ==="
  echo "[menu] Custom + Torch on CIFAR-10, the fused CIFAR CNN, MNIST, ECG200, and tabular sets."
  echo "[menu] Inference, unit tests, and micro-benchmarks are skipped. Failures do not abort."
  echo "[menu] Completed jobs are skipped until results/run_all_progress.json is deleted."
  echo
  begin_full_run run_all_training

  echo "[menu] -- Training (Tabular) --"
  run_optional train_tabular_custom tabular_demo
  run_optional train_tabular_torch tabular_demo
  run_optional train_tabular_custom tabular_iris
  run_optional train_tabular_torch tabular_iris
  run_optional train_tabular_custom tabular_wisconsin
  run_optional train_tabular_torch tabular_wisconsin

  echo "[menu] -- Training (MNIST) --"
  run_optional train_mnist_custom
  run_optional train_mnist_torch

  echo "[menu] -- Training (CIFAR-10) --"
  run_optional train_cifar_custom
  run_optional train_cifar_torch

  echo "[menu] -- Training (CIFAR-10 fused CBR) --"
  run_optional train_fused_custom
  run_optional train_fused_torch

  echo "[menu] -- Training (ECG200) --"
  run_optional train_gru_custom ecg200
  run_optional train_gru_torch ecg200

  echo "[menu] -- Metrics plots --"
  run_plots_optional

  echo
  echo "[menu] Full training pipelines finished."
  finish_full_run
}

run_all() {
  local enable_resume="${1:-1}"
  echo
  echo "[menu] === Run All Pipelines & Generate Plots ==="
  echo "[menu] Each target is compiled on demand (dev.sh only builds dllib_tests)."
  echo "[menu] Torch binaries are skipped if LibTorch was not configured. Failures do not abort the sequence."
  if [[ "${enable_resume}" == "1" ]]; then
    echo "[menu] Completed jobs are skipped until results/run_all_progress.json is deleted."
  fi
  echo
  if [[ "${enable_resume}" == "1" ]]; then
    begin_full_run run_all
  fi

  echo "[menu] -- Training (Tabular) --"
  run_optional train_tabular_custom tabular_demo
  run_optional train_tabular_torch tabular_demo
  run_optional train_tabular_custom tabular_iris
  run_optional train_tabular_torch tabular_iris
  run_optional train_tabular_custom tabular_wisconsin
  run_optional train_tabular_torch tabular_wisconsin

  echo "[menu] -- Training (MNIST) --"
  run_optional train_mnist_custom
  run_optional train_mnist_torch

  echo "[menu] -- Training (CIFAR-10) --"
  run_optional train_cifar_custom
  run_optional train_cifar_torch

  echo "[menu] -- Training (CIFAR-10 fused CBR) --"
  run_optional train_fused_custom
  run_optional train_fused_torch

  echo "[menu] -- Training (ECG200) --"
  run_optional train_gru_custom ecg200
  run_optional train_gru_torch ecg200

  echo "[menu] -- Benchmarks --"
  run_optional bench_forward_custom
  run_optional bench_forward_torch
  run_optional bench_micro_ops --benchmark_min_time=0.5s --benchmark_counters_tabular=true

  echo "[menu] -- Metrics plots --"
  run_plots_optional

  echo
  echo "[menu] Run All finished."
  if [[ "${enable_resume}" == "1" ]]; then
    finish_full_run
  fi
}

run_sanity_check() {
  local previous="${EXPERIMENTS_JSON}"
  local sanity="${ROOT}/config/sanity.json"
  restore_experiments_json() {
    export EXPERIMENTS_JSON="${previous:-${DEFAULT_EXPERIMENTS_JSON}}"
    echo "[menu] Restored EXPERIMENTS_JSON=${EXPERIMENTS_JSON}"
    trap - RETURN INT TERM
  }

  if [[ ! -f "${sanity}" ]]; then
    echo "[menu] Sanity config not found: ${sanity}" >&2
    return 1
  fi

  trap restore_experiments_json RETURN INT TERM
  echo
  echo "[menu] === Sanity Check (fast end-to-end via sanity.json) ==="
  echo "[menu] Overriding EXPERIMENTS_JSON=${sanity}"
  echo "[menu] Pipelines load this file through load_pipeline_config(); C++ is unchanged."
  export EXPERIMENTS_JSON="${sanity}"
  run_optional train_tabular_custom tabular_demo
  run_optional train_gru_custom ecg200
  run_all 0
}

print_menu() {
  cat <<'EOF'

DeepLearnLib
  --- Setup ---
  0)  Setup datasets (download & generate)
  1)  Run unit tests

  --- Training (CIFAR-10) ---
  2)  Train custom CNN on CIFAR-10
  3)  Train Torch CNN on CIFAR-10

  --- Training (MNIST) ---
  4)  Train custom CNN on MNIST
  5)  Train Torch CNN on MNIST

  --- Training (Tabular) ---
  6)  Train custom MLP (demo)
  7)  Train Torch MLP (demo)
  8)  Train custom MLP (Iris)
  9)  Train Torch MLP (Iris)
  10) Train custom MLP (Wisconsin)
  11) Train Torch MLP (Wisconsin)

  --- Training (ECG200) ---
  12) Train custom GRU on ECG200
  13) Train Torch GRU on ECG200

  --- Training (CIFAR-10 fused CBR) ---
  14) Train custom fused CNN on CIFAR-10
  15) Train Torch fused CNN on CIFAR-10

  --- Benchmarks ---
  16) Forward latency (custom)
  17) Forward latency (Torch)
  18) Micro-benchmarks (custom vs Torch ops)

  --- Reports ---
  19) Plot metrics (CSV → PNG)
  20) Run all pipelines & generate plots (skips jobs in run_all_progress.json)
  21) Run all full training pipelines (skips jobs in run_all_progress.json)
  22) Sanity Check (fast end-to-end via sanity.json)
  23) Exit

EOF
}

while true; do
  print_menu
  read -r -p "Select [0-23]: " choice
  case "${choice}" in
    0) run_setup_datasets || true ;;
    1) run_bin dllib_tests || true ;;
    2) run_bin train_cifar_custom || true ;;
    3) run_bin train_cifar_torch || true ;;
    4) run_bin train_mnist_custom || true ;;
    5) run_bin train_mnist_torch || true ;;
    6) run_bin train_tabular_custom tabular_demo || true ;;
    7) run_bin train_tabular_torch tabular_demo || true ;;
    8) run_bin train_tabular_custom tabular_iris || true ;;
    9) run_bin train_tabular_torch tabular_iris || true ;;
    10) run_bin train_tabular_custom tabular_wisconsin || true ;;
    11) run_bin train_tabular_torch tabular_wisconsin || true ;;
    12) run_bin train_gru_custom ecg200 || true ;;
    13) run_bin train_gru_torch ecg200 || true ;;
    14) run_bin train_fused_custom || true ;;
    15) run_bin train_fused_torch || true ;;
    16) run_bin bench_forward_custom || true ;;
    17) run_bin bench_forward_torch || true ;;
    18) run_bin bench_micro_ops --benchmark_min_time=0.5s --benchmark_counters_tabular=true || true ;;
    19) run_plots || true ;;
    20) run_all ;;
    21) run_all_training ;;
    22) run_sanity_check ;;
    23)
      echo "[menu] Bye."
      exit 0
      ;;
    *)
      echo "[menu] Unknown option: ${choice}"
      ;;
  esac
done
