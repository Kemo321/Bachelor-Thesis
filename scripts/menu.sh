#!/usr/bin/env bash
# Interactive launcher for DeepLearnLib binaries (Docker and local Linux).
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${ROOT}/build}"
DEFAULT_EXPERIMENTS_JSON="${ROOT}/config/experiments.json"
export EXPERIMENTS_JSON="${EXPERIMENTS_JSON:-${DEFAULT_EXPERIMENTS_JSON}}"

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

has_bin() {
  find_bin "$1" >/dev/null
}

run_bin() {
  local name="$1"
  shift || true
  local bin
  if ! bin="$(find_bin "${name}")"; then
    echo "[menu] Binary '${name}' not found under ${BUILD_DIR}." >&2
    echo "[menu] Build first (./scripts/dev.sh or cmake --build ${BUILD_DIR})." >&2
    return 1
  fi
  echo "[menu] Running ${bin} $*"
  (cd "${BUILD_DIR}" && "${bin}" "$@")
}

# Used by Run All: skip missing targets (especially Torch) and keep going on failure.
run_optional() {
  local name="$1"
  shift || true
  if ! has_bin "${name}"; then
    echo "[menu] Skipping '${name}' (binary not found; Torch baselines require LibTorch)."
    return 0
  fi
  if ! run_bin "${name}" "$@"; then
    echo "[menu] WARNING: '${name}' failed; continuing."
    return 0
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

run_plots() {
  local py
  if ! py="$(python_bin)"; then
    echo "[menu] Python not found; cannot run plot_metrics.py." >&2
    return 1
  fi
  echo "[menu] Running ${ROOT}/scripts/plot_metrics.py"
  "${py}" "${ROOT}/scripts/plot_metrics.py" --results-root "${ROOT}/results"
}

run_all() {
  echo
  echo "[menu] === Run All Pipelines & Generate Plots ==="
  echo "[menu] Missing Torch binaries are skipped. Failures do not abort the sequence."
  echo

  echo "[menu] -- Training: VOC --"
  run_optional train_pipeline_custom
  run_optional train_pipeline_torch

  echo "[menu] -- Training: BCCD --"
  run_optional train_pipeline_custom_BCCD
  run_optional train_pipeline_torch_BCCD

  echo "[menu] -- Training: Synthetic --"
  run_optional train_pipeline_custom_synthetic
  run_optional train_pipeline_torch_synthetic

  echo "[menu] -- Inference benchmarks --"
  run_optional inference_bccd
  run_optional inference_synthetic
  run_optional bench_custom --benchmark_min_time=0.1
  run_optional bench_torch --benchmark_min_time=0.1

  echo "[menu] -- Metrics plots --"
  run_plots || echo "[menu] WARNING: plotting failed; continuing."

  echo
  echo "[menu] Run All finished."
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
  run_optional tabular_demo
  run_all
}

print_menu() {
  cat <<'EOF'

DeepLearnLib
  1)  Run unit tests
  2)  Train custom YOLO on VOC
  3)  Train Torch YOLO on VOC
  4)  Train custom YOLO on BCCD
  5)  Train Torch YOLO on BCCD
  6)  Train custom YOLO on Synthetic
  7)  Train Torch YOLO on Synthetic
  8)  Run tabular demo
  9)  Run custom performance benchmarks
  10) Run BCCD inference
  11) Run Synthetic inference
  12) Plot metrics (CSV → PNG)
  13) Run all pipelines & generate plots
  14) Run Sanity Check (fast end-to-end test using sanity.json)
  15) Exit

EOF
}

while true; do
  print_menu
  read -r -p "Select [1-15]: " choice
  case "${choice}" in
    1)
      run_bin dllib_tests || true
      ;;
    2)
      run_bin train_pipeline_custom || true
      ;;
    3)
      run_bin train_pipeline_torch || true
      ;;
    4)
      run_bin train_pipeline_custom_BCCD || true
      ;;
    5)
      run_bin train_pipeline_torch_BCCD || true
      ;;
    6)
      run_bin train_pipeline_custom_synthetic || true
      ;;
    7)
      run_bin train_pipeline_torch_synthetic || true
      ;;
    8)
      run_bin tabular_demo || true
      ;;
    9)
      run_bin bench_custom --benchmark_min_time=0.1 || true
      ;;
    10)
      run_bin inference_bccd || true
      ;;
    11)
      run_bin inference_synthetic || true
      ;;
    12)
      run_plots || true
      ;;
    13)
      run_all
      ;;
    14)
      run_sanity_check
      ;;
    15)
      echo "[menu] Bye."
      exit 0
      ;;
    *)
      echo "[menu] Unknown option: ${choice}"
      ;;
  esac
done
