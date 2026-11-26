#!/usr/bin/env bash
# Starts an N-rank tensor-parallel rack on this machine. Ctrl-C stops it.
#
#   scripts/run_local_tp.sh weights/gpt2.mrkw 2
#   scripts/run_local_tp.sh weights/gpt2.mrkw 2 weights/moe8.mrkw    # Mixture-of-Experts mode
#
# Then, from another terminal:
#   python3 -c "import sys; sys.path.insert(0,'scripts'); from mr_client import Client; \
#               print(Client('127.0.0.1', 9100).forward([15496, 11, 616, 3290, 318], last_only=True).argmax())"
set -euo pipefail

WEIGHTS="${1:?usage: scripts/run_local_tp.sh path/to/gpt2.mrkw [nranks] [moe-file]}"
N="${2:-2}"
MOE="${3:-}"
MOE_ARGS=()
if [[ -n "$MOE" ]]; then MOE_ARGS=(--moe "$MOE"); fi
BIN="$(dirname "$0")/../build/mr_rank"

trap 'kill 0' EXIT

"$BIN" --rank 0 --nranks "$N" --weights "$WEIGHTS" "${MOE_ARGS[@]}" --port 9100 &
for ((r = 1; r < N; r++)); do
  "$BIN" --rank "$r" --nranks "$N" --weights "$WEIGHTS" "${MOE_ARGS[@]}" --root 127.0.0.1:9100 &
done
wait
