#!/usr/bin/env bash
# Local GPU check. GitHub-hosted runners have no NVIDIA GPU, and this script
# is not a CI job. CI only compiles dllib_tests and the custom binaries.
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"

if [[ -n "${GITHUB_ACTIONS:-}" ]]; then
  echo "Refusing to run under GitHub Actions. Hosted runners have no NVIDIA GPU."
  exit 1
fi

if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi failed. This script is for a local machine with a GPU, not CI."
  exit 1
fi

run_short=0
if [[ "${1:-}" == "--short" ]]; then
  run_short=1
elif [[ -n "${1:-}" ]]; then
  echo "Usage: scripts/verify_gpu.sh [--short]"
  exit 1
fi

test_bin=""
for candidate in \
  build/tests/dllib_tests \
  build/tests/dllib_tests.exe \
  build_release/tests/dllib_tests \
  build_release/tests/dllib_tests.exe \
  build_debug/tests/dllib_tests \
  build_debug/tests/dllib_tests.exe
do
  if [[ -f "$candidate" ]]; then
    test_bin="$candidate"
    break
  fi
done

if [[ -z "$test_bin" ]]; then
  echo "dllib_tests was not found. Build first with scripts/dev.sh or scripts/dev.ps1."
  exit 1
fi

echo "Running $test_bin"
"$test_bin"

if [[ "$run_short" -eq 0 ]]; then
  exit 0
fi

voc_root="$root/data/VOCdevkit/VOC2012"
if [[ ! -d "$voc_root" ]]; then
  echo "Skipping short_voc_custom: $voc_root is not present."
  exit 0
fi

short_bin=""
for candidate in \
  build/benchmarks/short_voc_custom \
  build/benchmarks/short_voc_custom.exe \
  build_release/benchmarks/short_voc_custom \
  build_release/benchmarks/short_voc_custom.exe \
  build_debug/benchmarks/short_voc_custom \
  build_debug/benchmarks/short_voc_custom.exe
do
  if [[ -f "$candidate" ]]; then
    short_bin="$candidate"
    break
  fi
done

if [[ -z "$short_bin" ]]; then
  echo "short_voc_custom was not built. Tests passed; the short run was skipped."
  exit 0
fi

echo "Running $short_bin (3 epochs)"
"$short_bin"
