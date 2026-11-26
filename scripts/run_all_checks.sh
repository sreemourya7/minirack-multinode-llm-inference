#!/usr/bin/env bash
# Builds everything and runs every check, Parts 1 to 6. Takes several minutes.
#
#   scripts/run_all_checks.sh weights/gpt2.mrkw weights/gpt2
#
# Arguments: the converted weight file, and the folder holding GPT-2's
# config.json and model.safetensors (see the README's quick start).
set -euo pipefail

WEIGHTS="${1:?usage: scripts/run_all_checks.sh path/to/gpt2.mrkw path/to/gpt2-folder}"
HF_DIR="${2:?usage: scripts/run_all_checks.sh path/to/gpt2.mrkw path/to/gpt2-folder}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

cmake -S "$ROOT" -B "$ROOT/build" > /dev/null
cmake --build "$ROOT/build" -j

failed=0
for part in 1 2 3 4 5 6; do
  echo
  echo "================ Part $part ================"
  if ! python3 "$ROOT/tests/check_part$part.py" --weights "$WEIGHTS" --hf-dir "$HF_DIR"; then
    failed=1
  fi
done

echo
if [[ $failed -ne 0 ]]; then
  echo "SOME CHECKS FAILED"
  exit 1
fi
echo "ALL CHECKS PASSED (Parts 1 to 6)"
