#!/usr/bin/env python3
"""Part 6 check: AVX2 kernels, the profile, and the memory roofline.

    python3 tests/check_part6.py --weights gpt2.mrkw --hf-dir /path/to/gpt2

A. Kernel unit tests (build/mr_kernel_test): the AVX2 kernels agree with the
   scalar ones, and every kernel is batch-invariant.
B. End to end, for each kernel set: logits match PyTorch, cached decoding is
   bit-identical to recomputing, greedy text equals Hugging Face's. The MoE
   model still makes every routing decision the same way.
C. A rack refuses a rank that uses different kernels.
D. Microbenchmark (build/mr_bench): kernel speed hot and cold, memory read speed.
E. Before and after: prefill and decode, scalar against AVX2, with the profile
   of where the time goes.
F. The roofline: decode time against the time needed just to read the weights,
   at the read speed a rank really gets when the other ranks are reading too.

Saves build/logs/part6_profile.csv for scripts/plot_profile.py.
(Skips the AVX2 half on a CPU without AVX2 and FMA.)
"""
import argparse
import csv
import os
import re
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import (ROOT, TpRack, free_ports, hf_greedy, moe_reference_logits, reference_logits,  # noqa: E402
                    test_sequences)

from mr_client import Client  # noqa: E402  (path set up by common)

HELLO = [15496, 11, 616, 3290, 318, 13779]
ROUTER_SEED = 8
PHASES = ["matmul_ms", "head_ms", "gelu_ms", "attention_ms", "norm_ms"]
MODEL_MB = 494.6  # weights a dense decode step reads: every block matrix plus the token table
failures = 0


def check(ok, label, detail=""):
    global failures
    failures += not ok
    print(f"  {label}: {'ok' if ok else 'FAILED   <-- FAIL'}{('  (' + detail + ')') if detail else ''}", flush=True)
    return ok


def mean(rows, key):
    return sum(r[key] for r in rows) / len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--hf-dir", required=True)
    ap.add_argument("--build-dir", default=os.path.join(ROOT, "build"))
    ap.add_argument("--log-dir", default=os.path.join(ROOT, "build", "logs"))
    ap.add_argument("--tol", type=float, default=2e-3)
    args = ap.parse_args()
    os.makedirs(args.log_dir, exist_ok=True)
    rank_bin = os.path.join(args.build_dir, "mr_rank")

    # ------------------------------------------------------------- A. kernel tests
    print("A. kernel unit tests")
    out = subprocess.run([os.path.join(args.build_dir, "mr_kernel_test")], capture_output=True, text=True)
    has_avx2 = "AVX2 + FMA available: yes" in out.stdout
    check(out.returncode == 0, "mr_kernel_test", out.stdout.strip().splitlines()[-1])
    kernels = ["scalar", "avx2"] if has_avx2 else ["scalar"]
    if not has_avx2:
        print("  this CPU has no AVX2 + FMA: only the scalar kernels are exercised below")

    def rack(n, kern, tag, moe=None):
        return TpRack(rank_bin, args.weights, n, args.log_dir, tag=f"_p6{tag}_{kern}", moe=moe, env={"MR_KERNELS": kern})

    # ---------------------------------------------------------- B. end to end
    print("\nB. end to end with each kernel set (2 ranks)")
    sequences = test_sequences()
    seq32 = next(s for s in sequences if len(s) == 32)
    refs, _, _ = reference_logits(args.hf_dir, sequences)
    want = hf_greedy(args.hf_dir, HELLO, 24)
    moe_file = os.path.join(args.log_dir, "moe_e8_k2_softmax.mrkw")
    subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "make_moe.py"), "--base", args.weights, "--out",
                    moe_file, "--experts", "8", "--top-k", "2", "--seed", str(ROUTER_SEED)],
                   check=True, stdout=subprocess.DEVNULL)
    moe_refs, moe_loads, _, (_, top_k) = moe_reference_logits(args.hf_dir, moe_file, sequences)
    for kern in kernels:
        with rack(2, kern, "B") as r, Client("127.0.0.1", r.port) as c:
            worst = max(float(np.abs(c.forward(ids) - ref).max()) for ids, ref in zip(sequences, refs))
            cached = [c.prefill(1, seq32[:20])] + [c.decode(1, t) for t in seq32[20:28]]
            fresh = [c.forward(seq32[:20 + i], last_only=True)[0] for i in range(9)]
            c.release(1)
            identical = all(np.array_equal(a, b) for a, b in zip(cached, fresh))
            text_ok = c.generate(HELLO, 24) == want
            check(worst < args.tol and identical and text_ok, f"{kern}: dense",
                  f"max |diff| vs PyTorch {worst:.1e}; cached == recomputed; greedy text == Hugging Face")
        with rack(2, kern, "B_moe", moe=moe_file) as r, Client("127.0.0.1", r.port) as c:
            worst, routing = 0.0, True
            for ids, ref, load in zip(sequences, moe_refs, moe_loads):
                worst = max(worst, float(np.abs(c.forward(ids) - ref).max()))
                time.sleep(0.05)
                routing &= r.stats(0)[-1]["expert_load"] == load
            check(worst < args.tol and routing, f"{kern}: MoE", f"max |diff| vs PyTorch MoE {worst:.1e}; every routing decision agrees")

    # ---------------------------------------------- C. mixed kernels are refused
    if has_avx2:
        print("\nC. a rank with different kernels tries to join")
        (port,) = free_ports(1)
        logs = [os.path.join(args.log_dir, f"p6_mixed_rank{k}.log") for k in (0, 1)]
        base = [rank_bin, "--nranks", "2", "--weights", args.weights]
        root = subprocess.Popen(base + ["--rank", "0", "--port", str(port)], stderr=open(logs[0], "w"),
                                env=dict(os.environ, MR_KERNELS="avx2"))
        odd = subprocess.Popen(base + ["--rank", "1", "--root", f"127.0.0.1:{port}"], stderr=open(logs[1], "w"),
                               env=dict(os.environ, MR_KERNELS="scalar"))
        try:
            odd.wait(timeout=30)
            exited = True
        except subprocess.TimeoutExpired:
            exited = False
        time.sleep(0.2)
        turned_away = "turned away a rank" in open(logs[0]).read()
        root.terminate()
        odd.kill()
        root.wait()
        check(exited and turned_away, "scalar rank refused by an AVX2 root, and it exits",
              "a replacement must reproduce the lost rank's arithmetic exactly")

    # ------------------------------------------------------------- D. microbenchmark
    print("\nD. microbenchmark (build/mr_bench)")
    bench = subprocess.run([os.path.join(args.build_dir, "mr_bench")], capture_output=True, text=True, check=True).stdout
    table = bench.split("memory read_gb_s")[0]
    print("\n".join("  " + line for line in table.strip().splitlines()[1:]))
    read_gb_s = float(re.search(r"memory read_gb_s=([\d.]+)", bench).group(1))
    gemm = {m.group(1): {k: float(v) for k, v in re.findall(r"(\w+)=([\d.]+)", m.group(2))}
            for m in re.finditer(r"gemm kernel=(\w+) (.*)", bench)}
    # Weights read per second by EACH of n processes reading at the same time.
    read_each = {int(m.group(1)): float(m.group(2))
                 for m in re.finditer(r"concurrent procs=(\d+) each_gb_s=([\d.]+)", bench)}

    # ---------------------------------------------------------- E. before and after
    print("\nE. before and after (32-token prompt, then 32 decoded tokens; all times in ms)")
    results = {}
    configs = [("dense", 1, None), ("dense", 2, None), ("MoE", 2, moe_file)]
    for label, n, moe in configs:
        for kern in kernels:
            with rack(n, kern, f"E_{label}", moe=moe) as r, Client("127.0.0.1", r.port) as c:
                c.forward(seq32, last_only=True)  # warm up
                before = len(r.stats(0))
                for sid in (1, 2, 3):
                    c.prefill(sid, seq32, argmax=True)
                    c.release(sid)
                token, _ = c.prefill(4, seq32, argmax=True)
                for _ in range(32):
                    token, _ = c.decode(4, token, argmax=True)
                c.release(4)
                long_prompt = (seq32 * 8)[:256]
                t0 = time.perf_counter()
                c.prefill(5, long_prompt, argmax=True)
                long_ms = (time.perf_counter() - t0) * 1e3
                c.release(5)
                time.sleep(0.3)
                per_rank = [r.stats(k)[before:] for k in range(n)]
                pre = [[x for x in rows if x["op"] == "prefill" and x["T"] == 32] for rows in per_rank]
                best = min(range(3), key=lambda j: pre[0][j]["total_ms"])
                dec = [[x for x in rows if x["op"] == "decode"] for rows in per_rank]
                results[(label, n, kern)] = {
                    "prefill": pre[0][best]["total_ms"],
                    "prefill_phases": {p: pre[0][best][p] for p in PHASES},
                    "prefill_compute": max(rows[best]["compute_ms"] for rows in pre),
                    "prefill_copy": min(rows[best]["collective_ms"] for rows in pre),
                    "decode": mean(dec[0], "total_ms"),
                    "decode_phases": {p: mean(dec[0], p) for p in PHASES},
                    "decode_compute": max(mean(rows, "compute_ms") for rows in dec),
                    "decode_copy": min(mean(rows, "collective_ms") for rows in dec),
                    "long_tokens_per_s": 256 / long_ms * 1e3,
                }
    print(f"  {'model':<7}{'ranks':>6}{'kernels':>9}{'prefill 32':>12}{'per token':>11}{'decode/token':>14}{'256-token prefill':>20}")
    for label, n, _ in configs:
        for kern in kernels:
            x = results[(label, n, kern)]
            print(f"  {label:<7}{n:>6}{kern:>9}{x['prefill']:>12.0f}{x['prefill'] / 32:>11.1f}{x['decode']:>14.1f}"
                  f"{x['long_tokens_per_s']:>13.0f} tok/s")
        if has_avx2:
            s, a = results[(label, n, "scalar")], results[(label, n, "avx2")]
            print(f"  {'':<7}{'':>6}{'speedup':>9}{s['prefill'] / a['prefill']:>11.1f}x{'':>11}{s['decode'] / a['decode']:>13.1f}x"
                  f"{a['long_tokens_per_s'] / s['long_tokens_per_s']:>19.1f}x")

    print("\n  where one rank's time goes, dense, 1 rank (ms per token)")
    print(f"  {'':<22}{'matmul':>9}{'head':>8}{'GELU':>8}{'attention':>11}{'layernorm':>11}{'other':>8}{'total':>9}")
    profile_rows = []
    for stage, per in [("prefill", 32), ("decode", 1)]:
        for kern in kernels:
            x = results[("dense", 1, kern)]
            ph = {p: v / per for p, v in x[f"{stage}_phases"].items()}
            total = x[stage] / per
            other = total - sum(ph.values())
            print(f"  {stage + ', ' + kern:<22}{ph['matmul_ms']:>9.2f}{ph['head_ms']:>8.2f}{ph['gelu_ms']:>8.2f}"
                  f"{ph['attention_ms']:>11.2f}{ph['norm_ms']:>11.2f}{other:>8.2f}{total:>9.2f}")
            profile_rows.append([stage, kern, f"{ph['matmul_ms']:.3f}", f"{ph['head_ms']:.3f}", f"{ph['gelu_ms']:.3f}",
                                 f"{ph['attention_ms'] + ph['norm_ms'] + other:.3f}", f"{total:.3f}"])

    print("\n  compute against communication, dense, 2 ranks (ms per request)")
    print(f"  {'':<22}{'total':>9}{'slowest rank computes':>24}{'copying messages':>19}{'the rest: waiting':>20}")
    for stage in ["prefill", "decode"]:
        for kern in kernels:
            x = results[("dense", 2, kern)]
            total, comp, copy = x[stage], x[f"{stage}_compute"], x[f"{stage}_copy"]
            print(f"  {stage + ', ' + kern:<22}{total:>9.1f}{comp:>24.1f}{copy:>13.2f} ({100 * copy / total:>2.0f}%)"
                  f"{max(total - comp - copy, 0.0):>20.1f}")

    # ------------------------------------------------------------------ F. roofline
    # The read speed a rank can count on is what the microbenchmark measured
    # with that many processes reading weights at once: on one machine all
    # ranks share the same memory.
    best_kern = kernels[-1]
    single = read_each.get(1) or gemm[best_kern]["cold1_gb_s"]
    print("\nF. the memory roofline for decode")
    print(f"  a process reading weights alone gets {single:.1f} GB/s"
          + (f"; with two at once each gets {read_each[2]:.1f} GB/s" if 2 in read_each else ""))
    print(f"  {'model':<7}{'ranks':>6}{'weights read per':>18}{'read speed per':>16}{'time just to':>14}{'measured compute,':>19}{'measured /':>12}")
    print(f"  {'':<7}{'':>6}{'token per rank, MB':>18}{'rank, GB/s':>16}{'read them, ms':>14}{'slowest rank, ms':>19}{'roofline':>12}")
    roofline_ms = None
    # MoE reads attention's matrices, the token table, and only the chosen experts: k of E of the MLP.
    moe_mb = 113.4 + 154.4 + 226.5 * top_k / 8
    for label, n, total_mb in [("dense", 1, MODEL_MB), ("dense", 2, MODEL_MB), ("MoE", 2, moe_mb)]:
        per_rank_mb = total_mb / n
        rate = read_each.get(n, single)
        floor = per_rank_mb / 1e3 / rate * 1e3
        measured = results[(label, n, best_kern)]["decode_compute"]
        if (label, n) == ("dense", 1):
            roofline_ms = floor
        note = "  (average; the split depends on routing)" if label == "MoE" else ""
        print(f"  {label:<7}{n:>6}{per_rank_mb:>18.0f}{rate:>16.1f}{floor:>14.1f}{measured:>19.1f}{measured / floor:>11.2f}x{note}")
    if has_avx2:
        g = gemm["avx2"]
        print(f"  kernel alone: {g['hot32_gmac']:.1f} billion multiply-adds/s with the weights in cache, "
              f"{g['cold1_gmac']:.1f} when each weight is read once")
    d1 = results[("dense", 1, best_kern)]["decode"]
    d2 = results[("dense", 2, best_kern)]["decode"]
    if 2 in read_each:
        print(f"  a second rank made decode {d1 / d2:.2f}x faster; the memory measurement predicts about "
              f"{2 * read_each[2] / single:.2f}x (two processes read {2 * read_each[2]:.1f} GB/s together, one reads {single:.1f})")

    csv_path = os.path.join(args.log_dir, "part6_profile.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["stage", "kernels", "matmul_ms", "head_ms", "gelu_ms", "other_ms", "total_ms", "roofline_ms"])
        for row in profile_rows:
            w.writerow(row + [f"{roofline_ms:.3f}" if row[0] == "decode" else ""])
    print(f"\n(profile saved to {os.path.relpath(csv_path, ROOT)})")

    print()
    if failures:
        print(f"FAILED: {failures} check(s)")
        sys.exit(1)
    print("PASSED: the AVX2 kernels are correct end to end, and the measurements are above")


if __name__ == "__main__":
    main()
