#!/usr/bin/env bash
# Interactive launcher for DeepLearnLib binaries (Docker and local Linux).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${ROOT}/build}"
export EXPERIMENTS_JSON="${EXPERIMENTS_JSON:-${ROOT}/config/experiments.json}"

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
  local bin
  if ! bin="$(find_bin "${name}")"; then
    echo "[menu] Binary '${name}' not found under ${BUILD_DIR}." >&2
    echo "[menu] Build first (./scripts/dev.sh or cmake --build ${BUILD_DIR})." >&2
    return 1
  fi
  echo "[menu] Running ${bin} $*"
  (cd "${BUILD_DIR}" && "${bin}" "$@")
}

print_menu() {
  cat <<'EOF'

DeepLearnLib
  1) Run unit tests
  2) Train custom YOLO on VOC
  3) Run tabular demo
  4) Run performance benchmarks
  5) Exit

EOF
}

while true; do
  print_menu
  read -r -p "Select [1-5]: " choice
  case "${choice}" in
    1)
      run_bin dllib_tests
      ;;
    2)
      run_bin train_pipeline_custom
      ;;
    3)
      run_bin tabular_demo
      ;;
    4)
      run_bin bench_custom --benchmark_min_time=0.1
      ;;
    5)
      echo "[menu] Bye."
      exit 0
      ;;
    *)
      echo "[menu] Unknown option: ${choice}"
      ;;
  esac
done
