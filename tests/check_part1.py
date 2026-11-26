#!/usr/bin/env python3
"""Part 1 correctness check: minirack vs the Hugging Face GPT-2 reference.

    python3 tests/check_part1.py --weights gpt2.mrkw --hf-dir /path/to/gpt2

--hf-dir is a folder holding GPT-2's config.json and model.safetensors.

For several ways of splitting the 12 layers across workers, this script
starts the workers, sends token sequences through the chain, and checks:

  1. logits match PyTorch at every position (max abs difference under --tol)
  2. the predicted next token (argmax) matches at every position
  3. every split gives bit-identical output to the single-worker run
     (splitting the model must not change the arithmetic at all)
  4. last_only returns exactly the final row
  5. bad requests get an error reply and the workers keep serving
"""
import argparse
import os
import socket
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from mr_client import Client, WorkerError  # noqa: E402

# Each entry: list of (first_layer, last_layer) per worker, in chain order.
SPLITS = {
    "1 worker  (0-12)": [(0, 12)],
    "2 workers (6 | 6)": [(0, 6), (6, 12)],
    "2 workers (1 | 11)": [(0, 1), (1, 12)],
    "3 workers (4 | 4 | 4)": [(0, 4), (4, 8), (8, 12)],
}
LENGTHS = [1, 2, 7, 16, 32]


def free_ports(n):
    socks = [socket.socket() for _ in range(n)]
    for s in socks:
        s.bind(("127.0.0.1", 0))
    ports = [s.getsockname()[1] for s in socks]
    for s in socks:
        s.close()
    return ports


def wait_for_port(port, proc, timeout=60.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"worker on port {port} exited early (code {proc.returncode})")
        try:
            socket.create_connection(("127.0.0.1", port), timeout=0.5).close()
            return
        except OSError:
            time.sleep(0.1)
    raise RuntimeError(f"worker on port {port} did not come up")


class Rack:
    """Starts a chain of workers and stops them on exit."""

    def __init__(self, binary, weights, layer_ranges, log_dir):
        self.procs = []
        ports = free_ports(len(layer_ranges))
        self.port = ports[0]
        n = len(layer_ranges)
        # Start from the back so each worker's --next is already listening.
        for i in reversed(range(n)):
            a, b = layer_ranges[i]
            cmd = [binary, "--port", str(ports[i]), "--weights", weights, "--layers", f"{a}:{b}"]
            if i == 0:
                cmd.append("--embed")
            if i == n - 1:
                cmd.append("--head")
            else:
                cmd += ["--next", f"127.0.0.1:{ports[i + 1]}"]
            split = "_".join(f"{x}-{y}" for x, y in layer_ranges)
            log = open(os.path.join(log_dir, f"split_{split}_worker{i}.log"), "w")
            proc = subprocess.Popen(cmd, stderr=log, stdout=log)
            self.procs.append(proc)
            wait_for_port(ports[i], proc)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        for p in self.procs:
            p.terminate()
        for p in self.procs:
            p.wait(timeout=10)


def reference_logits(hf_dir, sequences):
    import torch
    from transformers import GPT2LMHeadModel

    torch.set_grad_enabled(False)
    model = GPT2LMHeadModel.from_pretrained(hf_dir, local_files_only=True).eval()
    out = []
    for ids in sequences:
        logits = model(torch.tensor([ids], dtype=torch.long)).logits[0]
        out.append(logits.numpy().astype(np.float32))
    return out, model.config.vocab_size, model.config.n_positions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help=".mrkw file from scripts/export_weights.py")
    ap.add_argument("--hf-dir", required=True, help="folder with GPT-2 config.json + model.safetensors")
    ap.add_argument("--worker", default=os.path.join(ROOT, "build", "mr_worker"))
    ap.add_argument("--tol", type=float, default=2e-3, help="max allowed abs logit difference vs PyTorch")
    ap.add_argument("--log-dir", default=os.path.join(ROOT, "build", "logs"))
    args = ap.parse_args()
    os.makedirs(args.log_dir, exist_ok=True)

    rng = np.random.default_rng(0)
    sequences = [rng.integers(0, 50257, size=n).tolist() for n in LENGTHS]
    # "Hello, my dog is cute" in GPT-2 tokens: a real sentence alongside the random ones.
    sequences.append([15496, 11, 616, 3290, 318, 13779])

    print("computing PyTorch reference ...", flush=True)
    refs, vocab, n_ctx = reference_logits(args.hf_dir, sequences)

    failures = 0
    baseline = None  # outputs of the single-worker run
    print(f"\n{'split':<24}{'T':>4}{'max |diff|':>14}{'argmax':>9}{'== 1-worker':>13}{'ms':>9}")
    for name, ranges in SPLITS.items():
        with Rack(args.worker, args.weights, ranges, args.log_dir) as rack, Client("127.0.0.1", rack.port) as c:
            outputs = []
            for ids, ref in zip(sequences, refs):
                t0 = time.perf_counter()
                got = c.forward(ids)
                ms = (time.perf_counter() - t0) * 1e3
                outputs.append(got.copy())
                diff = float(np.abs(got - ref).max())
                argmax_ok = bool((got.argmax(-1) == ref.argmax(-1)).all())
                same = "-" if baseline is None else ("yes" if np.array_equal(got, baseline[len(outputs) - 1]) else "NO")
                ok = diff < args.tol and argmax_ok and same != "NO"
                failures += not ok
                print(f"{name:<24}{len(ids):>4}{diff:>14.2e}{'ok' if argmax_ok else 'FAIL':>9}{same:>13}{ms:>9.0f}"
                      + ("" if ok else "   <-- FAIL"))

            # last_only must be exactly the final row of the full answer.
            last = c.forward(sequences[3], last_only=True)
            if last.shape != (1, vocab) or not np.array_equal(last[0], outputs[3][-1]):
                failures += 1
                print(f"{name:<24} last_only mismatch   <-- FAIL")

            # Bad requests: expect an error reply, then a normal answer afterwards.
            for label, bad in [("token id out of range", [vocab]), ("negative token id", [-1]),
                               ("empty sequence", []), ("longer than context", [0] * (n_ctx + 1))]:
                try:
                    c.forward(bad)
                    failures += 1
                    print(f"{name:<24} {label}: no error raised   <-- FAIL")
                except WorkerError:
                    pass
            again = c.forward(sequences[0])
            if not np.array_equal(again, outputs[0]):
                failures += 1
                print(f"{name:<24} wrong answer after error requests   <-- FAIL")

            if baseline is None:
                baseline = outputs

    print()
    if failures:
        print(f"FAILED: {failures} check(s)")
        sys.exit(1)
    print("PASSED: all splits match PyTorch, match each other bit for bit, and survive bad requests")


if __name__ == "__main__":
    main()
