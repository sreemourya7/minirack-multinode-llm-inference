#!/usr/bin/env python3
"""Part 4 correctness and speed check: sessions with a KV cache.

    python3 tests/check_part4.py --weights gpt2.mrkw --hf-dir /path/to/gpt2

A. The cache changes nothing. Logits from prefill + step-by-step decode must
   be bit-identical to re-running the whole sequence with no cache, and must
   match PyTorch. Dense on 1, 2 and 4 ranks, and MoE on 2 ranks.

B. Sessions are isolated. Two sessions advanced in alternation give exactly
   what each gives alone.

C. Lifecycle. Wrong uses get an error and leave everything intact; a client
   that disconnects without releasing has its sessions freed for it; the
   session limit is enforced.

D. Greedy generation matches Hugging Face `generate`, token for token.

E. Timing: prefill, decode, the cost of the same token with no cache, how
   much data each decode step moves, and the size of the cache.
"""
import argparse
import os
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import (ROOT, TpRack, detokenize, hf_greedy, moe_reference_logits, reference_logits,  # noqa: E402
                    test_sequences)

from mr_client import Client, WorkerError  # noqa: E402  (path set up by common)

PROMPT_LEN = 12     # section A: tokens given to prefill
DECODE_STEPS = 12   # section A: tokens then fed one at a time
ROUTER_SEED = 8     # same MoE router as Part 3 (see check_part3.py for why)
HELLO = [15496, 11, 616, 3290, 318, 13779]  # "Hello, my dog is cute"
N_GENERATE = 24


def expect_error(label, fn, contains=""):
    """Returns 0 if fn raises a WorkerError (mentioning `contains`), else 1."""
    try:
        fn()
    except WorkerError as e:
        if contains in str(e):
            print(f"  {label}: error reply ({e})")
            return 0
        print(f"  {label}: wrong error ({e})   <-- FAIL")
        return 1
    print(f"  {label}: no error   <-- FAIL")
    return 1


def mean(rows, key):
    return sum(r[key] for r in rows) / len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help=".mrkw file from scripts/export_weights.py")
    ap.add_argument("--hf-dir", required=True, help="folder with GPT-2 config.json + model.safetensors")
    ap.add_argument("--rank-binary", default=os.path.join(ROOT, "build", "mr_rank"))
    ap.add_argument("--tol", type=float, default=2e-3, help="max allowed abs logit difference vs PyTorch")
    ap.add_argument("--log-dir", default=os.path.join(ROOT, "build", "logs"))
    args = ap.parse_args()
    os.makedirs(args.log_dir, exist_ok=True)

    moe_file = os.path.join(args.log_dir, "moe_e8_k2_softmax.mrkw")
    subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "make_moe.py"), "--base", args.weights, "--out",
                    moe_file, "--experts", "8", "--top-k", "2", "--seed", str(ROUTER_SEED)],
                   check=True, stdout=subprocess.DEVNULL)

    seq32 = next(s for s in test_sequences() if len(s) == 32)
    seq = seq32[:PROMPT_LEN + DECODE_STEPS]
    print("computing PyTorch references ...", flush=True)
    (dense_ref,), vocab, n_ctx = reference_logits(args.hf_dir, [seq])
    (moe_ref,), _, _, _ = moe_reference_logits(args.hf_dir, moe_file, [seq])
    failures = 0

    def rack(n, tag, moe=None):
        return TpRack(args.rank_binary, args.weights, n, args.log_dir, tag=tag, moe=moe)

    # ------------------------------------------------- A. the cache changes nothing
    print(f"\nA. prefill {PROMPT_LEN} tokens, then decode {DECODE_STEPS} one at a time")
    print(f"{'model':<8}{'ranks':>6}{'steps == no-cache run':>24}{'max |diff| vs PyTorch':>24}{'argmax on rack':>17}")
    for label, n, moe, ref in [("dense", 1, None, dense_ref), ("dense", 2, None, dense_ref),
                               ("dense", 4, None, dense_ref), ("MoE", 2, moe_file, moe_ref)]:
        with rack(n, "_kv", moe) as r, Client("127.0.0.1", r.port) as c:
            identical, worst = 0, 0.0
            for i in range(PROMPT_LEN - 1, len(seq) - 1):
                # Position i: the logits after seeing tokens 0..i.
                got = c.prefill(1, seq[:PROMPT_LEN]) if i == PROMPT_LEN - 1 else c.decode(1, seq[i])
                fresh = c.forward(seq[:i + 1], last_only=True)[0]
                identical += bool(np.array_equal(got, fresh))
                worst = max(worst, float(np.abs(got - ref[i]).max()))
            # Final step with the token chosen on the rack.
            token, logit = c.decode(1, seq[-1], argmax=True)
            fresh = c.forward(seq, last_only=True)[0]
            argmax_ok = token == int(fresh.argmax()) and logit == float(fresh.max())
            c.release(1)
            ok = identical == DECODE_STEPS and worst < args.tol and argmax_ok
            failures += not ok
            print(f"{label:<8}{n:>6}{f'{identical} of {DECODE_STEPS} bit-identical':>24}{worst:>24.2e}"
                  f"{'same token' if argmax_ok else 'DIFFERS':>17}" + ("" if ok else "   <-- FAIL"))

    with rack(2, "_kv_misc") as r:
        # ----------------------------------------------------- B. session isolation
        print("\nB. two sessions advanced in alternation (2 ranks)")
        a, b = seq32[:10], seq32[16:26]
        with Client("127.0.0.1", r.port) as c:
            def run_alone(sid, s):
                out = [c.prefill(sid, s[:6])] + [c.decode(sid, t) for t in s[6:]]
                c.release(sid)
                return out
            alone_a, alone_b = run_alone(1, a), run_alone(2, b)
            mixed_a, mixed_b = [c.prefill(1, a[:6])], [c.prefill(2, b[:6])]
            for ta, tb in zip(a[6:], b[6:]):
                mixed_a.append(c.decode(1, ta))
                mixed_b.append(c.decode(2, tb))
            same = all(np.array_equal(x, y) for x, y in zip(alone_a + alone_b, mixed_a + mixed_b))
            failures += not same
            print(f"  interleaved output vs each session alone: {'bit-identical' if same else 'DIFFERS   <-- FAIL'}")

            # ------------------------------------------------------------ C. lifecycle
            print("\nC. lifecycle (sessions 1 and 2 are still open)")
            failures += expect_error("decode on a session that was never opened", lambda: c.decode(77, 1), "no such session")
            failures += expect_error("prefill on a session id already in use", lambda: c.prefill(1, a[:3]), "already exists")
            failures += expect_error("release of an unknown session", lambda: c.release(77), "no such session")
            failures += expect_error("prefill longer than the context window", lambda: c.prefill(5, [0] * (n_ctx + 1)), "context")
            failures += expect_error("decode with an out-of-range token", lambda: c.decode(1, vocab), "out of range")
            # None of those may have disturbed session 1: its next step must still be right.
            nxt = c.decode(1, seq32[10])
            intact = np.array_equal(nxt, c.forward(a + [seq32[10]], last_only=True)[0])
            failures += not intact
            print(f"  session 1 after all those errors: {'intact' if intact else 'DAMAGED   <-- FAIL'}")
            c.release(1)
            failures += expect_error("decode after release", lambda: c.decode(1, 1), "no such session")
            # Session 2 is deliberately left open; the client now disconnects.

        with Client("127.0.0.1", r.port) as c:
            try:
                c.prefill(2, b[:3])  # only works if the rack freed session 2 itself
                print("  session left open by a client that disconnected: freed by the rack")
            except WorkerError as e:
                failures += 1
                print(f"  session left open by a client that disconnected: still there ({e})   <-- FAIL")
            c.release(2)
            limit = 64
            for sid in range(100, 100 + limit):
                c.prefill(sid, [0])
            failures += expect_error(f"opening session number {limit + 1}", lambda: c.prefill(999, [0]), "too many")
            for sid in range(100, 100 + limit):
                c.release(sid)

        # ------------------------------------------------------ D. greedy generation
        print(f"\nD. greedy generation, {N_GENERATE} new tokens (2 ranks) vs Hugging Face generate")
        want = hf_greedy(args.hf_dir, HELLO, N_GENERATE)
        with Client("127.0.0.1", r.port) as c:
            got = c.generate(HELLO, N_GENERATE)
        same = got == want
        failures += not same
        print(f"  tokens: {'identical' if same else 'DIFFER   <-- FAIL'}")
        text = detokenize(args.hf_dir, HELLO + got)
        if text:
            print(f"  text:   {text!r}")

    # ------------------------------------------------------------------ E. timing
    print("\nE. timing: 32-token prompt, then 32 decoded tokens")
    print(f"{'model':<8}{'ranks':>6}{'prefill':>10}{'decode':>10}{'of which':>10}{'in collec-':>12}{'data per':>10}"
          f"{'same token,':>13}{'cache at':>12}")
    print(f"{'':<8}{'':>6}{'ms':>10}{'ms/token':>10}{'head ms':>10}{'tives, ms':>12}{'step, kB':>10}"
          f"{'no cache ms':>13}{'64 tok, MB':>12}")
    summary = {}
    for label, n, moe in [("dense", 1, None), ("dense", 2, None), ("MoE", 2, moe_file)]:
        with rack(n, "_kv_timing", moe) as r, Client("127.0.0.1", r.port) as c:
            before = len(r.stats(0))
            token, _ = c.prefill(1, seq32, argmax=True)
            for _ in range(32):
                token, _ = c.decode(1, token, argmax=True)
            c.release(1)
            fresh = [c.forward(seq32 + seq32, last_only=True) for _ in range(2)]  # 64 tokens, no cache
            assert fresh
            time.sleep(0.3)
            per_rank = [r.stats(k)[before:] for k in range(n)]
            root = per_rank[0]
            prefill = next(x for x in root if x["op"] == "prefill")
            decodes = [x for x in root if x["op"] == "decode"]
            no_cache = min(x["total_ms"] for x in root if x["op"] == "forward")
            last_decode = [[x for x in rows if x["op"] == "decode"][-1] for rows in per_rank]
            cache_mb = sum(x["kv_kb"] for x in last_decode) * 64 / last_decode[0]["len"] / 1e3
            d_ms, head = mean(decodes, "total_ms"), mean(decodes, "head_ms")
            coll, kb = mean(decodes, "collective_ms"), mean(decodes, "sent_kb") + mean(decodes, "recv_kb")
            summary[(label, n)] = (prefill["total_ms"], d_ms, no_cache)
            print(f"{label:<8}{n:>6}{prefill['total_ms']:>10.0f}{d_ms:>10.1f}{head:>10.1f}{coll:>12.2f}{kb:>10.1f}"
                  f"{no_cache:>13.0f}{cache_mb:>12.2f}")

    # Decode with the full logits shipped to the root instead of an (logit, id) pair.
    with rack(2, "_kv_timing_logits") as r, Client("127.0.0.1", r.port) as c:
        before = len(r.stats(0))
        logits = c.prefill(1, seq32)
        for _ in range(32):
            logits = c.decode(1, int(logits.argmax()))
        c.release(1)
        time.sleep(0.3)
        decodes = [x for x in r.stats(0)[before:] if x["op"] == "decode"]
        kb_logits = mean(decodes, "sent_kb") + mean(decodes, "recv_kb")
        ms_logits = mean(decodes, "total_ms")

    p1, d1, nc1 = summary[("dense", 1)]
    p2, d2, nc2 = summary[("dense", 2)]
    print(f"\ndense, 1 rank: decoding token 65 takes {d1:.0f} ms with the cache and {nc1:.0f} ms without ({nc1 / d1:.0f}x)")
    print(f"dense, 2 ranks: prefill {32 / p2 * 1e3:.0f} tokens/s, decode {1e3 / d2:.1f} tokens/s; "
          f"2 ranks vs 1: prefill {p1 / p2:.2f}x, decode {d1 / d2:.2f}x")
    print(f"dense, 2 ranks, shipping full logits to the root instead of picking the token on the rack: "
          f"{kb_logits:.0f} kB per step, {ms_logits:.1f} ms/token")

    print()
    if failures:
        print(f"FAILED: {failures} check(s)")
        sys.exit(1)
    print("PASSED: cached decoding is bit-identical to recomputing, sessions are isolated, and generation matches Hugging Face")


if __name__ == "__main__":
    main()
