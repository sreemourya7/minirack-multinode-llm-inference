#!/usr/bin/env python3
"""Part 2 correctness and speed check: the tensor-parallel rack vs PyTorch.

    python3 tests/check_part2.py --weights gpt2.mrkw --hf-dir /path/to/gpt2

For 1, 2, 3 and 4 ranks this script starts the rack and checks:

  1. logits match PyTorch at every position (max abs difference under --tol)
  2. the predicted next token (argmax) matches at every position
  3. the same request twice gives bit-identical output (determinism)
  4. last_only returns exactly the final row
  5. bad requests get an error reply and the rack keeps serving

Every request also runs the rack's own check that all ranks hold a
bit-identical hidden state. Two failure paths are then forced on purpose:

  6. one rank corrupts its hidden state  -> the replica check must catch it
  7. one rank is killed                  -> the client gets an error, not a hang

Finally it times a 32-token request for each rank count.
"""
import argparse
import os
import re
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import ROOT, free_ports, reference_logits, test_sequences

from mr_client import Client, WorkerError  # noqa: E402  (path set up by common)

RANK_COUNTS = [1, 2, 3, 4]
TIMED_T = 32
TIMED_RUNS = 5


class TpRack:
    """Starts N ranks (root first) and stops them on exit."""

    def __init__(self, binary, weights, nranks, log_dir, tag="", env=None):
        self.procs = []
        self.logs = []
        (self.port,) = free_ports(1)
        for r in range(nranks):
            cmd = [binary, "--rank", str(r), "--nranks", str(nranks), "--weights", weights]
            cmd += ["--port", str(self.port)] if r == 0 else ["--root", f"127.0.0.1:{self.port}"]
            path = os.path.join(log_dir, f"tp{tag}_{nranks}ranks_rank{r}.log")
            self.logs.append(path)
            full_env = dict(os.environ, **(env or {}))
            self.procs.append(subprocess.Popen(cmd, stderr=open(path, "w"), stdout=subprocess.DEVNULL, env=full_env))
        self._wait_ready()

    def _wait_ready(self, timeout=120.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            for p in self.procs:
                if p.poll() is not None:
                    raise RuntimeError(f"a rank exited early (code {p.returncode}); see the logs")
            try:
                with Client("127.0.0.1", self.port, timeout=5.0) as c:
                    if c.ping():
                        return
            except OSError:
                time.sleep(0.2)
        raise RuntimeError("the rack did not become ready")

    def root_stats(self):
        """Parses the root's per-request log lines into dicts."""
        rows = []
        with open(self.logs[0]) as f:
            for line in f:
                if " req " in line and "total_ms" in line:
                    rows.append({k: float(v) for k, v in re.findall(r"(\w+)=([\d.]+)", line)})
        return rows

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        for p in self.procs:
            p.terminate()
        for p in self.procs:
            p.wait(timeout=10)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help=".mrkw file from scripts/export_weights.py")
    ap.add_argument("--hf-dir", required=True, help="folder with GPT-2 config.json + model.safetensors")
    ap.add_argument("--rank-binary", default=os.path.join(ROOT, "build", "mr_rank"))
    ap.add_argument("--tol", type=float, default=2e-3, help="max allowed abs logit difference vs PyTorch")
    ap.add_argument("--log-dir", default=os.path.join(ROOT, "build", "logs"))
    args = ap.parse_args()
    os.makedirs(args.log_dir, exist_ok=True)

    sequences = test_sequences()
    print("computing PyTorch reference ...", flush=True)
    refs, vocab, n_ctx = reference_logits(args.hf_dir, sequences)
    timed_ids = next(s for s in sequences if len(s) == TIMED_T)

    failures = 0
    timing = []  # (nranks, best total ms, compute ms, collective ms, kB moved by root, collectives)
    print(f"\n{'ranks':<7}{'T':>4}{'max |diff|':>14}{'argmax':>9}{'repeatable':>12}")
    for n in RANK_COUNTS:
        with TpRack(args.rank_binary, args.weights, n, args.log_dir) as rack, Client("127.0.0.1", rack.port) as c:
            outputs = []
            for ids, ref in zip(sequences, refs):
                got = c.forward(ids).copy()
                again = c.forward(ids)
                outputs.append(got)
                diff = float(np.abs(got - ref).max())
                argmax_ok = bool((got.argmax(-1) == ref.argmax(-1)).all())
                repeat_ok = bool(np.array_equal(got, again))
                ok = diff < args.tol and argmax_ok and repeat_ok
                failures += not ok
                print(f"{n:<7}{len(ids):>4}{diff:>14.2e}{'ok' if argmax_ok else 'FAIL':>9}"
                      f"{'yes' if repeat_ok else 'NO':>12}" + ("" if ok else "   <-- FAIL"))

            last = c.forward(sequences[3], last_only=True)
            if last.shape != (1, vocab) or not np.array_equal(last[0], outputs[3][-1]):
                failures += 1
                print(f"{n:<7} last_only mismatch   <-- FAIL")

            for label, bad in [("token id out of range", [vocab]), ("negative token id", [-1]),
                               ("empty sequence", []), ("longer than context", [0] * (n_ctx + 1))]:
                try:
                    c.forward(bad)
                    failures += 1
                    print(f"{n:<7} {label}: no error raised   <-- FAIL")
                except WorkerError:
                    pass
            if not np.array_equal(c.forward(sequences[0]), outputs[0]):
                failures += 1
                print(f"{n:<7} wrong answer after error requests   <-- FAIL")

            # Timing: keep the fastest of a few runs (least disturbed by other processes).
            before = len(rack.root_stats())
            for _ in range(TIMED_RUNS):
                c.forward(timed_ids, last_only=True)
            time.sleep(0.2)  # let the root flush its log
            runs = rack.root_stats()[before:]
            best = min(runs, key=lambda r: r["total_ms"])
            timing.append((n, best["total_ms"], best["compute_ms"], best["collective_ms"],
                           best["sent_kb"] + best["recv_kb"], int(best["collectives"])))

    # --- failure paths: the rack must answer with an error, never hang or lie ---
    print("\nfailure paths (2 ranks)")
    with TpRack(args.rank_binary, args.weights, 2, args.log_dir, tag="_corrupt",
                env={"MR_TEST_CORRUPT_RANK": "1"}) as rack, Client("127.0.0.1", rack.port) as c:
        try:
            c.forward(sequences[2])
            failures += 1
            print("  rank 1 corrupts its hidden state: NOT detected   <-- FAIL")
        except WorkerError as e:
            ok = "replica check failed" in str(e)
            failures += not ok
            print(f"  rank 1 corrupts its hidden state: {'detected' if ok else 'wrong error  <-- FAIL'} ({e})")

    with TpRack(args.rank_binary, args.weights, 2, args.log_dir, tag="_killed") as rack, \
            Client("127.0.0.1", rack.port, timeout=30.0) as c:
        c.forward(sequences[2])  # healthy before the kill
        rack.procs[1].kill()
        rack.procs[1].wait()
        for attempt in (1, 2):
            try:
                c.forward(sequences[2])
                failures += 1
                print(f"  rank 1 killed, request {attempt}: got an answer   <-- FAIL")
            except WorkerError as e:
                print(f"  rank 1 killed, request {attempt}: error reply ({e})")

    base = timing[0][1]
    print(f"\nlatency of one {TIMED_T}-token request (best of {TIMED_RUNS}), measured on the root rank")
    print(f"{'ranks':<7}{'total ms':>10}{'speedup':>9}{'root compute':>14}{'in collectives':>16}{'kB moved':>10}{'collectives':>13}")
    for n, total, compute, coll, kb, calls in timing:
        print(f"{n:<7}{total:>10.0f}{base / total:>8.2f}x{compute:>14.0f}{coll:>16.0f}{kb:>10.0f}{calls:>13}")
    print("('in collectives' includes waiting for slower ranks, not only copying bytes)")

    print()
    if failures:
        print(f"FAILED: {failures} check(s)")
        sys.exit(1)
    print("PASSED: every rank count matches PyTorch, is repeatable bit for bit, and survives bad requests")


if __name__ == "__main__":
    main()
