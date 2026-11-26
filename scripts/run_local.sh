#!/usr/bin/env bash
# Starts a 2-worker rack on this machine. Ctrl-C stops both workers.
#
#   scripts/run_local.sh weights/gpt2.mrkw
#
# Then, from another terminal:
#   python3 -c "import sys; sys.path.insert(0,'scripts'); from mr_client import Client; \
#               print(Client('127.0.0.1', 9000).forward([15496, 11, 616, 3290, 318], last_only=True).argmax())"
set -euo pipefail

WEIGHTS="${1:?usage: scripts/run_local.sh path/to/gpt2.mrkw}"
BIN="$(dirname "$0")/../build/mr_worker"

trap 'kill 0' EXIT

# Last worker first, so worker 0 has someone to connect to.
"$BIN" --port 9001 --weights "$WEIGHTS" --layers 6:12 --head &
"$BIN" --port 9000 --weights "$WEIGHTS" --layers 0:6 --embed --next 127.0.0.1:9001 &
wait
