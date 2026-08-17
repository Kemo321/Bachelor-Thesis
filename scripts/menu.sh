#!/usr/bin/env bash
# Interactive launcher for DeepLearnLib binaries (Docker and local Linux).
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${ROOT}/build}"
DEFAULT_EXPERIMENTS_JSON="${ROOT}/config/experiments.json"
export EXPERIMENTS_JSON="${EXPERIMENTS_JSON:-${DEFAULT_EXPERIMENTS_JSON}}"
# shellcheck source=cuda_env.sh
source "${ROOT}/scripts/cuda_env.sh"

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

has_bin() {
  find_bin "$1" >/dev/null
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

run_all() {
  echo
  echo "[menu] === Run All Pipelines & Generate Plots ==="
  echo "[menu] Missing Torch binaries are skipped. Failures do not abort the sequence."
  echo

  echo "[menu] -- Training (Tabular) --"
  run_optional train_tabular_custom
  run_optional train_tabular_torch

  echo "[menu] -- Training (CIFAR-10) --"
  run_optional train_cifar_custom
  run_optional train_cifar_torch

  echo "[menu] -- Training (VOC) --"
  run_optional train_voc_custom
  run_optional train_voc_torch

  echo "[menu] -- Training (BCCD) --"
  run_optional train_bccd_custom
  run_optional train_bccd_torch

  echo "[menu] -- Training (Synthetic) --"
  run_optional train_synthetic_custom
  run_optional train_synthetic_torch

  echo "[menu] -- Inference --"
  run_optional inference_voc_custom
  run_optional inference_voc_torch
  run_optional inference_bccd_custom
  run_optional inference_bccd_torch
  run_optional inference_synthetic_custom
  run_optional inference_synthetic_torch

  echo "[menu] -- Benchmarks --"
  run_optional bench_voc_custom --benchmark_min_time=0.1s
  run_optional bench_voc_torch --benchmark_min_time=0.1s
  run_optional bench_micro_ops --benchmark_min_time=0.5s --benchmark_counters_tabular=true

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
  run_optional train_tabular_custom
  run_optional overfit_voc_custom
  run_all
}

print_menu() {
  cat <<'EOF'

DeepLearnLib
  --- Setup ---
  0)  Setup datasets (download & generate)
  1)  Run unit tests

  --- Training (VOC) ---
  2)  Train custom YOLO on VOC
  3)  Train Torch YOLO on VOC
  4)  Short VOC custom (3 epochs)
  5)  Short VOC Torch (3 epochs)
  6)  Overfit custom (tiny VOC)
  7)  Overfit Torch (tiny VOC)

  --- Training (BCCD) ---
  8)  Train custom YOLO on BCCD
  9)  Train Torch YOLO on BCCD

  --- Training (Synthetic) ---
  10) Train custom YOLO on Synthetic
  11) Train Torch YOLO on Synthetic

  --- Training (CIFAR-10) ---
  12) Train custom CNN on CIFAR-10
  13) Train Torch CNN on CIFAR-10

  --- Training (Tabular) ---
  14) Train custom MLP
  15) Train Torch MLP

  --- Inference ---
  16) Infer VOC custom
  17) Infer VOC Torch
  18) Infer BCCD custom
  19) Infer BCCD Torch
  20) Infer Synthetic custom
  21) Infer Synthetic Torch

  --- Benchmarks ---
  22) Bench VOC custom
  23) Bench VOC Torch
  24) Micro-benchmarks (custom vs Torch ops)

  --- Reports ---
  25) Plot metrics (CSV → PNG)
  26) Run all pipelines & generate plots
  27) Sanity Check (fast end-to-end via sanity.json)
  28) Exit

EOF
}

while true; do
  print_menu
  read -r -p "Select [0-28]: " choice
  case "${choice}" in
    0) run_setup_datasets || true ;;
    1) run_bin dllib_tests || true ;;
    2) run_bin train_voc_custom || true ;;
    3) run_bin train_voc_torch || true ;;
    4) run_bin short_voc_custom || true ;;
    5) run_bin short_voc_torch || true ;;
    6) run_bin overfit_voc_custom || true ;;
    7) run_bin overfit_voc_torch || true ;;
    8) run_bin train_bccd_custom || true ;;
    9) run_bin train_bccd_torch || true ;;
    10) run_bin train_synthetic_custom || true ;;
    11) run_bin train_synthetic_torch || true ;;
    12) run_bin train_cifar_custom || true ;;
    13) run_bin train_cifar_torch || true ;;
    14) run_bin train_tabular_custom || true ;;
    15) run_bin train_tabular_torch || true ;;
    16) run_bin inference_voc_custom || true ;;
    17) run_bin inference_voc_torch || true ;;
    18) run_bin inference_bccd_custom || true ;;
    19) run_bin inference_bccd_torch || true ;;
    20) run_bin inference_synthetic_custom || true ;;
    21) run_bin inference_synthetic_torch || true ;;
    22) run_bin bench_voc_custom --benchmark_min_time=0.1s || true ;;
    23) run_bin bench_voc_torch --benchmark_min_time=0.1s || true ;;
    24) run_bin bench_micro_ops --benchmark_min_time=0.5s --benchmark_counters_tabular=true || true ;;
    25) run_plots || true ;;
    26) run_all ;;
    27) run_sanity_check ;;
    28)
      echo "[menu] Bye."
      exit 0
      ;;
    *)
      echo "[menu] Unknown option: ${choice}"
      ;;
  esac
done
