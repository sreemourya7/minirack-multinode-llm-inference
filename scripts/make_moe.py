#!/usr/bin/env python3
"""Create an MoE sidecar file: router weights and settings for mr_rank --moe.

    python3 scripts/make_moe.py --base weights/gpt2.mrkw --experts 8 --top-k 2 \
        --out weights/moe8.mrkw

The experts themselves are not stored here: expert e is hidden units
[e * U/E, (e+1) * U/E) of GPT-2's own MLP, read from the base weight file.
This file only adds what an MoE layer needs on top of that:

    moe.config               [1, 3]   n_expert, top_k, gate mode (1 softmax, 0 unit)
    h.L.moe.router.weight    [D, E]   one per layer
    h.L.moe.router.bias      [1, E]

IMPORTANT: the router is random, not trained. The resulting model does not
produce sensible text. It exists to exercise the runtime: routing, running
only the chosen experts, and the load each rank ends up with. With
--experts E --top-k E --gate unit every expert runs with weight 1, which is
exactly the original GPT-2 MLP; the tests use that as a known-correct anchor.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mrkw import read_mrkw, write_mrkw  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True, help="the GPT-2 .mrkw file (only its header is read)")
    ap.add_argument("--experts", type=int, default=8)
    ap.add_argument("--top-k", type=int, default=2)
    ap.add_argument("--gate", choices=["softmax", "unit"], default="softmax")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg, _ = read_mrkw(args.base, load_tensors=False)
    D, E, k = cfg["n_embd"], args.experts, args.top_k
    if (4 * D) % E != 0:
        raise SystemExit(f"--experts must divide the MLP width {4 * D}")
    if not 1 <= k <= E:
        raise SystemExit("--top-k must be between 1 and --experts")

    rng = np.random.default_rng(args.seed)
    tensors = [("moe.config", np.array([[E, k, 1.0 if args.gate == "softmax" else 0.0]], dtype=np.float32))]
    for layer in range(cfg["n_layer"]):
        # Scale 1/sqrt(D) gives router scores of order 1, so the softmax gates
        # are neither all-equal nor one-hot.
        w = rng.normal(0.0, 1.0 / np.sqrt(D), size=(D, E)).astype(np.float32)
        b = rng.normal(0.0, 0.1, size=(1, E)).astype(np.float32)
        tensors.append((f"h.{layer}.moe.router.weight", w))
        tensors.append((f"h.{layer}.moe.router.bias", b))

    size = write_mrkw(args.out, cfg, tensors)
    print(f"wrote {args.out}: {E} experts, top-{k}, {args.gate} gates, seed {args.seed}, {size / 1e3:.0f} kB")


if __name__ == "__main__":
    main()
