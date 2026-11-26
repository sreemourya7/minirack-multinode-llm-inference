#!/usr/bin/env python3
"""Part 3 correctness and speed check: the Mixture-of-Experts layer.

    python3 tests/check_part3.py --weights gpt2.mrkw --hf-dir /path/to/gpt2

A. Anchor. With 8 experts, top-8 and unit gates, every expert runs with
   weight 1. That is exactly GPT-2's own MLP, so the rack must match plain
   Hugging Face GPT-2. This proves the expert slicing and the gather/scatter
   path without trusting any reference of our own.

B. Real MoE. With 8 experts, top-2 and softmax gates, the rack must match a
   PyTorch MoE model (Hugging Face GPT-2 with each MLP replaced; see
   moe_reference_logits in tests/common.py). Checked for 1 to 4 ranks:
     - logits and next-token prediction match
     - the same request twice is bit-identical
     - tokens-per-expert reported by the rack equals the reference exactly
       (every routing decision agreed)
     - the (token, expert) pairs executed by all ranks add up to
       layers x tokens x k: nothing ran twice, nothing was skipped

C. Timing. Dense versus MoE on 1 and 2 ranks, with each rank's share of the
   expert work.
"""
import argparse
import os
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import ROOT, TpRack, moe_reference_logits, reference_logits, test_sequences  # noqa: E402

from mr_client import Client  # noqa: E402  (path set up by common)

TIMED_T = 32
TIMED_RUNS = 5
N_LAYER = 12

# The router is random, so its seed is a free choice. Routing is a hard
# decision: if the 2nd- and 3rd-best scores for some token are nearly equal,
# float noise between two correct implementations could pick different experts.
# Over 16 seeds the closest call on the test sequences ranged from 3e-6 to
# 5e-4. Seed 8 has a closest call of about 4e-4, far enough from float noise
# (around 1e-6 here) that the comparison is stable across machines.
ROUTER_SEED = 8
MIN_MARGIN = 1e-4


def make_moe(base, out, experts, top_k, gate, seed=0):
    subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "make_moe.py"), "--base", base, "--out", out,
                    "--experts", str(experts), "--top-k", str(top_k), "--gate", gate, "--seed", str(seed)],
                   check=True, stdout=subprocess.DEVNULL)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help=".mrkw file from scripts/export_weights.py")
    ap.add_argument("--hf-dir", required=True, help="folder with GPT-2 config.json + model.safetensors")
    ap.add_argument("--rank-binary", default=os.path.join(ROOT, "build", "mr_rank"))
    ap.add_argument("--tol", type=float, default=2e-3, help="max allowed abs logit difference vs PyTorch")
    ap.add_argument("--log-dir", default=os.path.join(ROOT, "build", "logs"))
    args = ap.parse_args()
    os.makedirs(args.log_dir, exist_ok=True)

    anchor_file = os.path.join(args.log_dir, "moe_anchor_e8_k8_unit.mrkw")
    moe_file = os.path.join(args.log_dir, "moe_e8_k2_softmax.mrkw")
    make_moe(args.weights, anchor_file, experts=8, top_k=8, gate="unit")
    make_moe(args.weights, moe_file, experts=8, top_k=2, gate="softmax", seed=ROUTER_SEED)

    sequences = test_sequences()
    timed_index = next(i for i, s in enumerate(sequences) if len(s) == TIMED_T)
    print("computing PyTorch references ...", flush=True)
    dense_refs, _, _ = reference_logits(args.hf_dir, sequences)
    moe_refs, ref_loads, margin, (E, k) = moe_reference_logits(args.hf_dir, moe_file, sequences)
    failures = 0

    # ---------------------------------------------------------------- A. anchor
    print("\nA. anchor: 8 experts, top-8, unit gates  ==  plain GPT-2 (Hugging Face)")
    print(f"{'ranks':<7}{'T':>4}{'max |diff|':>14}{'argmax':>9}")
    for n in [1, 2, 4]:
        with TpRack(args.rank_binary, args.weights, n, args.log_dir, tag="_moe_anchor", moe=anchor_file) as rack, \
                Client("127.0.0.1", rack.port) as c:
            for ids, ref in zip(sequences, dense_refs):
                got = c.forward(ids)
                diff = float(np.abs(got - ref).max())
                argmax_ok = bool((got.argmax(-1) == ref.argmax(-1)).all())
                ok = diff < args.tol and argmax_ok
                failures += not ok
                print(f"{n:<7}{len(ids):>4}{diff:>14.2e}{'ok' if argmax_ok else 'FAIL':>9}" + ("" if ok else "   <-- FAIL"))

    # -------------------------------------------------------------- B. real MoE
    print(f"\nB. MoE: {E} experts, top-{k}, softmax gates  ==  PyTorch MoE reference")
    print(f"   closest routing call in the reference: margin {margin:.2e} "
          f"({'well above float noise' if margin > MIN_MARGIN else 'TOO SMALL: this test fixture is ambiguous'})")
    failures += not margin > MIN_MARGIN
    print(f"{'ranks':<7}{'T':>4}{'max |diff|':>14}{'argmax':>9}{'repeatable':>12}{'routing':>10}{'pairs run':>11}")
    for n in [1, 2, 3, 4]:
        with TpRack(args.rank_binary, args.weights, n, args.log_dir, tag="_moe", moe=moe_file) as rack, \
                Client("127.0.0.1", rack.port) as c:
            for i, (ids, ref) in enumerate(zip(sequences, moe_refs)):
                got = c.forward(ids).copy()
                again = c.forward(ids)
                time.sleep(0.1)  # let every rank flush its log line
                diff = float(np.abs(got - ref).max())
                argmax_ok = bool((got.argmax(-1) == ref.argmax(-1)).all())
                repeat_ok = bool(np.array_equal(got, again))
                last = [rack.stats(r)[-1] for r in range(n)]
                routing_ok = last[0]["expert_load"] == ref_loads[i]
                pairs = int(sum(row["expert_rows"] for row in last))
                pairs_ok = pairs == N_LAYER * len(ids) * k
                ok = diff < args.tol and argmax_ok and repeat_ok and routing_ok and pairs_ok
                failures += not ok
                print(f"{n:<7}{len(ids):>4}{diff:>14.2e}{'ok' if argmax_ok else 'FAIL':>9}"
                      f"{'yes' if repeat_ok else 'NO':>12}{'same' if routing_ok else 'DIFFERS':>10}"
                      f"{pairs:>7}{'' if pairs_ok else ' !':<4}" + ("" if ok else "   <-- FAIL"))

    # ---------------------------------------------------------------- C. timing
    print(f"\nC. latency of one {TIMED_T}-token request (best of {TIMED_RUNS})")
    print(f"{'model':<18}{'ranks':>6}{'total ms':>10}{'per-rank compute ms':>22}{'per-rank expert work':>23}")
    timed_ids = sequences[timed_index]
    results = {}
    for label, moe in [("dense", None), (f"MoE top-{k} of {E}", moe_file)]:
        for n in [1, 2]:
            with TpRack(args.rank_binary, args.weights, n, args.log_dir, tag="_timing", moe=moe) as rack, \
                    Client("127.0.0.1", rack.port) as c:
                for _ in range(TIMED_RUNS):
                    c.forward(timed_ids, last_only=True)
                time.sleep(0.2)
                per_rank = [rack.stats(r)[-TIMED_RUNS:] for r in range(n)]
                best = min(range(TIMED_RUNS), key=lambda j: per_rank[0][j]["total_ms"])
                total = per_rank[0][best]["total_ms"]
                compute = " / ".join(f"{per_rank[r][best]['compute_ms']:.0f}" for r in range(n))
                rows = [per_rank[r][best]["expert_rows"] for r in range(n)]
                work = " / ".join(f"{100 * x / sum(rows):.0f}%" for x in rows) if moe else "-"
                results[(label, n)] = total
                print(f"{label:<18}{n:>6}{total:>10.0f}{compute:>22}{work:>23}")
    d1, d2 = results[("dense", 1)], results[("dense", 2)]
    m1, m2 = results[(f"MoE top-{k} of {E}", 1)], results[(f"MoE top-{k} of {E}", 2)]
    print(f"MoE vs dense on 1 rank: {d1 / m1:.2f}x faster.   MoE on 2 ranks vs 1: {m1 / m2:.2f}x.   "
          f"Dense on 2 ranks vs 1: {d1 / d2:.2f}x.")
    print(f"tokens routed to each expert for this request (all layers): {ref_loads[timed_index]}")

    print()
    if failures:
        print(f"FAILED: {failures} check(s)")
        sys.exit(1)
    print("PASSED: the anchor matches GPT-2, the MoE matches its PyTorch reference, and every routing decision agrees")


if __name__ == "__main__":
    main()
