#!/usr/bin/env python3
"""Part 5 check: a rank fails in the middle of generation, and nothing is lost.

    python3 tests/check_part5.py --weights gpt2.mrkw --hf-dir /path/to/gpt2

In every scenario the test first records what an undisturbed run produces,
then repeats it while making a rank fail, and requires the logits to be
bit-identical and the client to see no error.

A. A rank is killed between two steps (both recovery strategies, two open sessions).
B. A rank is killed in the middle of a request, with 2 and with 3 ranks. With
   3 ranks the survivors must drop the half-done request and fall back in step.
C. A rank hangs (SIGSTOP). Only a timeout can reveal that. The rank is
   replaced by a standby, and the hung process must exit by itself when it
   wakes up.
D. No replacement arrives. The client gets a clear error, nothing is lost,
   and work continues once a replacement does arrive.
E. Three failures during one generation.
F. End to end under the supervisor: generated text equals Hugging Face's.
G. The same with the Mixture-of-Experts model.

Then the measurements:
H1. What the client sees: the pause on the interrupted step, for a cold
    replacement or a standby, and for replay or checkpoint.
H2. Rebuild time against session length for the two strategies.
H3. What checkpointing costs on every step when nothing fails.
"""
import argparse
import csv
import os
import signal
import subprocess
import sys
import threading
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import ROOT, TpRack, detokenize, free_ports, hf_greedy, test_sequences  # noqa: E402

from mr_client import Client, WorkerError  # noqa: E402  (path set up by common)
from supervisor import Supervisor  # noqa: E402

HELLO = [15496, 11, 616, 3290, 318, 13779]  # "Hello, my dog is cute"
ROUTER_SEED = 8
LENGTHS = [16, 64, 128, 256, 512]  # H2: session lengths
failures = 0


def check(ok, label, detail=""):
    global failures
    failures += not ok
    print(f"  {label}: {'ok' if ok else 'FAILED   <-- FAIL'}{('  (' + detail + ')') if detail else ''}", flush=True)
    return ok


def story(c, sid, tokens, n_prompt, before_step=None):
    """Prefill tokens[:n_prompt], then decode the rest one at a time.
    before_step(i) runs just before step i is sent (step 0 is the prefill).
    Returns (logits per step, milliseconds per step)."""
    out, ms = [], []
    for i in range(len(tokens) - n_prompt + 1):
        if before_step:
            before_step(i)
        t0 = time.perf_counter()
        out.append(c.prefill(sid, tokens[:n_prompt]) if i == 0 else c.decode(sid, tokens[n_prompt + i - 1]))
        ms.append((time.perf_counter() - t0) * 1e3)
    return out, ms


def same(a, b):
    return len(a) == len(b) and all(np.array_equal(x, y) for x, y in zip(a, b))


def wait_log(rack, rank, text, count=1, timeout=30.0):
    """Waits until `text` has appeared `count` times in a rank's log."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        with open(rack.logs[rank]) as f:
            if f.read().count(text) >= count:
                return True
        time.sleep(0.02)
    return False


def kill_and_replace(rack, r):
    rack.kill(r)
    rack.start(r)


def later(delay_s, fn):
    """Runs fn on another thread after a delay (to strike in the middle of a request)."""
    t = threading.Thread(target=lambda: (time.sleep(delay_s), fn()), daemon=True)
    t.start()
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help=".mrkw file from scripts/export_weights.py")
    ap.add_argument("--hf-dir", required=True, help="folder with GPT-2 config.json + model.safetensors")
    ap.add_argument("--rank-binary", default=os.path.join(ROOT, "build", "mr_rank"))
    ap.add_argument("--log-dir", default=os.path.join(ROOT, "build", "logs"))
    ap.add_argument("--quick", action="store_true", help="skip the longest sessions in H2")
    args = ap.parse_args()
    os.makedirs(args.log_dir, exist_ok=True)

    seq32 = next(s for s in test_sequences() if len(s) == 32)
    a_tokens, b_tokens = seq32[:16], seq32[16:32]

    def rack(n, tag, strategy="replay", extra=(), moe=None):
        return TpRack(args.rank_binary, args.weights, n, args.log_dir, tag="_p5" + tag, moe=moe,
                      extra=["--recovery", strategy, *extra])

    # ------------------------------------------------ A. killed between two steps
    print("A. rank 1 killed between two steps, two sessions open (2 ranks)")
    for strategy in ["replay", "checkpoint"]:
        with rack(2, f"A_{strategy}", strategy) as r, Client("127.0.0.1", r.port) as c:
            base_a, _ = story(c, 1, a_tokens, 6)
            base_b, _ = story(c, 2, b_tokens, 6)
            c.release(1)
            c.release(2)
            got_a, got_b = [c.prefill(1, a_tokens[:6])], [c.prefill(2, b_tokens[:6])]
            for i, (ta, tb) in enumerate(zip(a_tokens[6:], b_tokens[6:])):
                if i == 4:
                    kill_and_replace(r, 1)
                got_a.append(c.decode(1, ta))
                got_b.append(c.decode(2, tb))
            rec = r.lines(0, "recovery strategy")
            check(same(base_a, got_a) and same(base_b, got_b) and len(rec) == 1 and rec[0]["sessions"] == 2,
                  f"{strategy}: both sessions continue bit-identically",
                  f"rebuilt {int(rec[0]['tokens'])} tokens in {rec[0]['rebuild_ms']:.0f} ms" if rec else "no recovery logged")

    # --------------------------------------------- B. killed in the middle of a request
    print("\nB. a rank killed in the middle of a request")
    for n, victim, strategy in [(2, 1, "replay"), (3, 2, "replay"), (3, 1, "checkpoint")]:
        with rack(n, f"B_{n}_{victim}", strategy) as r, Client("127.0.0.1", r.port) as c:
            base, _ = story(c, 1, seq32, 28)
            c.release(1)
            # Strike 150 ms into a prefill that takes several hundred ms.
            got, _ = story(c, 1, seq32, 28, before_step=lambda i: later(0.15, lambda: kill_and_replace(r, victim)) if i == 0 else None)
            fail = r.lines(0, "failure rank")
            mid = bool(fail) and fail[0].get("during") == "request"
            aborted = n == 2 or any("aborted by the root" in open(r.logs[k]).read() for k in range(1, n) if k != victim)
            check(same(base, got) and mid and aborted,
                  f"{n} ranks, rank {victim} killed mid-prefill, {strategy}: request re-run, output bit-identical",
                  "survivor was told to abort and fell back in step" if n == 3 and aborted else
                  ("" if mid else "the kill did not land mid-request"))
            # And once more in the middle of a single decode step (short, so best effort).
            c.release(1)
            hit = False
            for attempt in range(4):
                before = len(r.lines(0, "failure rank"))
                got, _ = story(c, 1, seq32, 28, before_step=lambda i: later(0.012, lambda: kill_and_replace(r, victim)) if i == 2 else None)
                c.release(1)
                ok = same(base, got)
                new = r.lines(0, "failure rank")[before:]
                hit = bool(new) and new[0].get("during") == "request"
                if not ok or hit:
                    break
            check(ok, f"{n} ranks, rank {victim} killed around a decode step, {strategy}: output bit-identical",
                  "the kill landed mid-request" if hit else "the kill landed between requests each time")

    # ------------------------------------------------------------------- C. a hang
    print("\nC. rank 1 hangs (SIGSTOP); the root's timeout is 1000 ms; a standby is waiting")
    with rack(2, "C_hang", "checkpoint", extra=["--rank-timeout-ms", "1000"]) as r, Client("127.0.0.1", r.port) as c:
        base, _ = story(c, 1, a_tokens, 6)
        c.release(1)
        r.start_standby(1)
        wait_log(r, 0, "standby for rank 1 is waiting")
        hung = r.procs[1]

        def freeze(i):
            if i == 5:
                hung.send_signal(signal.SIGSTOP)
        got, ms = story(c, 1, a_tokens, 6, before_step=freeze)
        fail = r.lines(0, "failure rank")
        check(same(base, got) and bool(fail) and fail[0].get("reason") == "timeout",
              "detected by timeout, standby took over, output bit-identical",
              f"that step took {ms[5]:.0f} ms instead of about {np.median(ms[1:5]):.0f}")
        hung.send_signal(signal.SIGCONT)  # the hung process wakes up...
        try:
            hung.wait(timeout=10)
            exited = True
        except subprocess.TimeoutExpired:
            exited = False
        still_fine = np.array_equal(c.decode(1, seq32[20]), c.forward(a_tokens + [seq32[20]], last_only=True)[0])
        check(exited and still_fine, "the old process, once woken, found itself shut out and exited; the rack was not disturbed")

    # ------------------------------------------------------- D. no replacement arrives
    print("\nD. rank 1 killed and nothing replaces it (the root waits 1500 ms)")
    with rack(2, "D_none", "replay", extra=["--rejoin-timeout-ms", "1500"]) as r, Client("127.0.0.1", r.port) as c:
        base, _ = story(c, 1, a_tokens, 6)
        c.release(1)
        got = [c.prefill(1, a_tokens[:6])] + [c.decode(1, t) for t in a_tokens[6:10]]
        r.kill(1)
        t0 = time.perf_counter()
        try:
            c.decode(1, a_tokens[10])
            check(False, "expected an error")
        except WorkerError as e:
            waited = (time.perf_counter() - t0) * 1e3
            check("degraded" in str(e) and 1300 < waited < 4000, "the client gets a clear error after the wait",
                  f"{waited:.0f} ms: {str(e)[:70]}...")
        r.start(1)  # help arrives late
        got += [c.decode(1, t) for t in a_tokens[10:]]
        check(same(base, got), "once a replacement arrives the session carries on, bit-identical; the failed step was not half-applied")

    # ------------------------------------------------ E. three failures, one generation
    print("\nE. rank 1 fails three times during one generation (checkpoint, standbys)")
    with rack(2, "E_three", "checkpoint") as r, Client("127.0.0.1", r.port) as c:
        base, _ = story(c, 1, seq32, 8)
        c.release(1)
        serving = [r.procs[1]]
        r.start_standby(1)
        wait_log(r, 0, "standby for rank 1 is waiting", 1)
        strikes = {5: 1, 12: 2, 19: 3}

        def strike(i):
            if i in strikes:
                serving[0].kill()
                serving[0].wait()
                serving[0] = r.spares[-1]
                r.start_standby(1)
                wait_log(r, 0, "standby for rank 1 is waiting", strikes[i] + 1)
        got, _ = story(c, 1, seq32, 8, before_step=strike)
        rec = r.lines(0, "recovery strategy")
        check(same(base, got) and len(rec) == 3, "output bit-identical after three failures", f"{len(rec)} recoveries logged")

    # ------------------------------------------------------- F. under the supervisor
    print("\nF. end to end: the supervisor replaces the rank; text vs Hugging Face generate")
    want = hf_greedy(args.hf_dir, HELLO, 24)
    (port,) = free_ports(1)
    sup_logs = os.path.join(args.log_dir, "p5_supervised")
    for f in ("rank0.log", "rank1.log"):
        if os.path.exists(os.path.join(sup_logs, f)):
            os.remove(os.path.join(sup_logs, f))
    events = []
    with Supervisor(args.rank_binary, args.weights, 2, port=port, root_args=["--recovery", "checkpoint"], standby=True,
                    log_dir=sup_logs, on_event=events.append) as sup, Client("127.0.0.1", port) as c:
        deadline = time.time() + 30
        while time.time() < deadline and "standby for rank 1 is waiting" not in open(os.path.join(sup_logs, "rank0.log")).read():
            time.sleep(0.05)
        token, _ = c.prefill(1, HELLO, argmax=True)
        got = [token]
        while len(got) < 24:
            if len(got) == 9:
                sup.kill(1)
            token, _ = c.decode(1, token, argmax=True)
            got.append(token)
        c.release(1)
        time.sleep(0.3)
    check(got == want and len(events) >= 1, "24 greedy tokens identical to Hugging Face despite the kill at token 9",
          events[0] if events else "the supervisor saw no failure")
    text = detokenize(args.hf_dir, HELLO + got)
    if text:
        print(f"    text: {text!r}")

    # ----------------------------------------------------------------------- G. MoE
    print("\nG. Mixture-of-Experts model, rank 1 killed between steps (replay)")
    moe_file = os.path.join(args.log_dir, "moe_e8_k2_softmax.mrkw")
    subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "make_moe.py"), "--base", args.weights, "--out",
                    moe_file, "--experts", "8", "--top-k", "2", "--seed", str(ROUTER_SEED)],
                   check=True, stdout=subprocess.DEVNULL)
    with rack(2, "G_moe", "replay", moe=moe_file) as r, Client("127.0.0.1", r.port) as c:
        base, _ = story(c, 1, a_tokens, 6)
        c.release(1)
        got, _ = story(c, 1, a_tokens, 6, before_step=lambda i: kill_and_replace(r, 1) if i == 6 else None)
        check(same(base, got), "output bit-identical")

    # ------------------------------------------------- H1. what the client sees
    print("\nH1. pause seen by the client on the step that hit the failure (64-token session, 2 ranks)")
    print(f"{'replacement':<14}{'state rebuilt by':<18}{'normal step':>12}{'that step':>11}{'got a rank':>12}{'rebuild':>9}")
    print(f"{'':<14}{'':<18}{'ms':>12}{'ms':>11}{'after, ms':>12}{'ms':>9}")
    long_tokens = (seq32 * 3)[:70]
    matrix = []
    for standby in [False, True]:
        for strategy in ["replay", "checkpoint"]:
            with rack(2, f"H1_{strategy}_{int(standby)}", strategy) as r, Client("127.0.0.1", r.port) as c:
                if standby:
                    r.start_standby(1)
                    wait_log(r, 0, "standby for rank 1 is waiting")

                def fail(i):
                    if i == 4:  # the session holds 62 + 3 = 65 tokens at this point
                        if standby:
                            r.kill(1)
                        else:
                            kill_and_replace(r, 1)
                _, ms = story(c, 1, long_tokens, 62, before_step=fail)
                time.sleep(0.2)
                rec = r.lines(0, "recovery strategy")[0]
                normal = float(np.median(ms[1:4] + ms[5:]))
                matrix.append((standby, strategy, normal, ms[4], rec["wait_rejoin_ms"], rec["rebuild_ms"]))
                print(f"{'standby' if standby else 'cold start':<14}{strategy:<18}{normal:>12.0f}{ms[4]:>11.0f}"
                      f"{rec['wait_rejoin_ms']:>12.0f}{rec['rebuild_ms']:>9.1f}")

    # ------------------------------------------ H2. rebuild time against session length
    lengths = LENGTHS[:-1] if args.quick else LENGTHS
    print("\nH2. time to rebuild one session on the replacement rank, by session length (2 ranks)")
    rebuild = {}
    for strategy in ["replay", "checkpoint"]:
        with rack(2, f"H2_{strategy}", strategy) as r, Client("127.0.0.1", r.port) as c:
            for L in lengths:
                tokens = (seq32 * 17)[:L]
                c.prefill(1, tokens[:L - 2])
                c.decode(1, tokens[L - 2])
                c.decode(1, tokens[L - 1])  # the session now holds exactly L tokens
                n_before = len(r.lines(0, "recovery strategy"))
                kill_and_replace(r, 1)
                c.decode(1, tokens[0])
                rec = r.lines(0, "recovery strategy")[n_before]
                rebuild[(strategy, L)] = (rec["rebuild_ms"], rec["restored_kb"])
                c.release(1)
    print(f"{'tokens':>8}{'replay, ms':>12}{'checkpoint, ms':>16}{'checkpoint copies, kB':>23}")
    for L in lengths:
        print(f"{L:>8}{rebuild[('replay', L)][0]:>12.0f}{rebuild[('checkpoint', L)][0]:>16.1f}{rebuild[('checkpoint', L)][1]:>23.0f}")
    csv_path = os.path.join(args.log_dir, "part5_recovery.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tokens", "replay_rebuild_ms", "checkpoint_rebuild_ms", "checkpoint_restored_kb"])
        for L in lengths:
            w.writerow([L, f"{rebuild[('replay', L)][0]:.1f}", f"{rebuild[('checkpoint', L)][0]:.2f}",
                        f"{rebuild[('checkpoint', L)][1]:.1f}"])
    print(f"(saved to {os.path.relpath(csv_path, ROOT)})")

    # ------------------------------------------- H3. the price of checkpointing
    print("\nH3. cost of checkpointing on every decode step when nothing fails (2 ranks, 32 steps)")
    print(f"{'strategy':<12}{'ms/token':>10}{'data per step, kB':>19}")
    steady = {}
    for strategy in ["replay", "checkpoint"]:
        with rack(2, f"H3_{strategy}", strategy) as r, Client("127.0.0.1", r.port) as c:
            token, _ = c.prefill(1, seq32, argmax=True)
            for _ in range(32):
                token, _ = c.decode(1, token, argmax=True)
            c.release(1)
            time.sleep(0.2)
            dec = [x for x in r.stats(0) if x["op"] == "decode"]
            ms = sum(x["total_ms"] for x in dec) / len(dec)
            kb = sum(x["sent_kb"] + x["recv_kb"] for x in dec) / len(dec)
            steady[strategy] = (ms, kb)
            print(f"{strategy:<12}{ms:>10.1f}{kb:>19.1f}")
    extra_kb = steady["checkpoint"][1] - steady["replay"][1]
    print(f"checkpointing adds {extra_kb:.0f} kB per step ({100 * extra_kb / steady['replay'][1]:.0f}% more data) and "
          f"the root keeps a second copy of the other ranks' cache: {extra_kb:.0f} kB per token")

    print()
    if failures:
        print(f"FAILED: {failures} check(s)")
        sys.exit(1)
    print("PASSED: after every kind of failure the output is bit-identical to an undisturbed run")


if __name__ == "__main__":
    main()
