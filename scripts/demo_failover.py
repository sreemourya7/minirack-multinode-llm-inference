#!/usr/bin/env python3
"""Kill a rank in the middle of generating text, and watch nothing go wrong.

    python3 scripts/demo_failover.py --weights weights/gpt2.mrkw --hf-dir weights/gpt2

What it does:
  1. starts a 2-rank rack under the supervisor
  2. generates text once, undisturbed
  3. generates the same text again, and part-way through kills rank 1 (kill -9)
  4. reports whether the two outputs are identical and how long the client waited

Things to try:
    --recovery replay          rebuild the lost cache by re-running the tokens (default: checkpoint)
    --no-standby               no spare process; the replacement has to load its weights first
    --hang                     freeze the rank (SIGSTOP) instead of killing it; found by timeout
    --kill-at 25               strike later, when there is more state to rebuild
"""
import argparse
import os
import re
import signal
import socket
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from mr_client import Client  # noqa: E402
from supervisor import Supervisor  # noqa: E402


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def generate(c, session, prompt_ids, n_new, show, strike_at=None, strike=None):
    """Greedy generation. Returns (tokens, milliseconds per step)."""
    tokens, ms = [], []
    token = None
    for i in range(n_new):
        if i == strike_at:
            strike()
        t0 = time.perf_counter()
        token, _ = c.prefill(session, prompt_ids, argmax=True) if i == 0 else c.decode(session, token, argmax=True)
        ms.append((time.perf_counter() - t0) * 1e3)
        tokens.append(token)
        show(token)
    c.release(session)
    return tokens, ms


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--hf-dir", help="folder with GPT-2's tokenizer files; without it token ids are printed")
    ap.add_argument("--prompt", default="The best way to keep a service running when a machine dies is")
    ap.add_argument("--tokens", type=int, default=36)
    ap.add_argument("--kill-at", type=int, default=14, help="which generated token the failure strikes before")
    ap.add_argument("--recovery", choices=["replay", "checkpoint"], default="checkpoint")
    ap.add_argument("--no-standby", action="store_true")
    ap.add_argument("--hang", action="store_true", help="freeze rank 1 instead of killing it")
    ap.add_argument("--rank-binary", default=os.path.join(HERE, "..", "build", "mr_rank"))
    ap.add_argument("--log-dir", default=os.path.join(HERE, "..", "build", "logs", "demo"))
    args = ap.parse_args()
    if args.hang and args.no_standby:
        raise SystemExit("--hang needs a standby: a frozen process never exits, so the supervisor would never replace it")

    tok = None
    if args.hf_dir:
        try:
            from transformers import GPT2TokenizerFast

            tok = GPT2TokenizerFast.from_pretrained(args.hf_dir, local_files_only=True)
        except Exception as e:  # tokenizer files missing, or transformers not installed
            print(f"(no tokenizer: {e}; printing token ids)", file=sys.stderr)
    prompt_ids = tok.encode(args.prompt) if tok else [15496, 11, 616, 3290, 318, 13779]

    def show(token):
        print(tok.decode([token]) if tok else f" {token}", end="", flush=True)

    port = free_port()
    for name in ("rank0.log", "rank1.log"):
        path = os.path.join(args.log_dir, name)
        if os.path.exists(path):
            os.remove(path)
    root_args = ["--recovery", args.recovery, "--rank-timeout-ms", "1000" if args.hang else "10000"]
    events = []
    print(f"starting a 2-rank rack (recovery: {args.recovery}, standby: {'no' if args.no_standby else 'yes'}) ...")
    with Supervisor(args.rank_binary, args.weights, 2, port=port, root_args=root_args, standby=not args.no_standby,
                    log_dir=args.log_dir, on_event=events.append) as sup, Client("127.0.0.1", port) as c:
        root_log = os.path.join(args.log_dir, "rank0.log")
        if not args.no_standby:  # give the spare time to load its weights and report in
            deadline = time.time() + 30
            while time.time() < deadline and "standby for rank 1 is waiting" not in open(root_log).read():
                time.sleep(0.05)

        print("\n--- run 1: undisturbed ---")
        print(args.prompt if tok else "", end="", flush=True)
        base, base_ms = generate(c, 1, prompt_ids, args.tokens, show)
        print()

        victim = sup.procs[1]
        how = "frozen (SIGSTOP)" if args.hang else "killed (kill -9)"

        def strike():
            print(f" [rank 1 {how}] ", end="", flush=True)
            sup.kill(1, signal.SIGSTOP if args.hang else signal.SIGKILL)

        print(f"\n--- run 2: rank 1 is {how} before token {args.kill_at} ---")
        print(args.prompt if tok else "", end="", flush=True)
        got, got_ms = generate(c, 2, prompt_ids, args.tokens, show, strike_at=args.kill_at, strike=strike)
        print()
        if args.hang:
            victim.send_signal(signal.SIGCONT)  # let the frozen process wake up; it should exit by itself
            try:
                victim.wait(timeout=10)
            except Exception:
                pass
        time.sleep(0.3)
        log = open(root_log).read()

    normal = statistics.median(got_ms[1:args.kill_at] + got_ms[args.kill_at + 1:])
    print("\n--- result ---")
    print(f"same tokens as the undisturbed run:  {'YES, all ' + str(len(got)) + ' identical' if got == base else 'NO'}")
    print(f"errors seen by the client:           none")
    print(f"a normal step:                       {normal:.0f} ms")
    print(f"the step that hit the failure:       {got_ms[args.kill_at]:.0f} ms")
    for line in re.findall(r"\] (failure .*|recovery .*|rank 1 replaced by its standby|rank 1 joined)", log):
        print(f"root log:                            {line}")
    for e in events:
        print(f"supervisor:                          {e}")
    if args.hang:
        print(f"the frozen process, once woken:      {'exited by itself' if victim.poll() is not None else 'STILL RUNNING'}")
    sys.exit(0 if got == base else 1)


if __name__ == "__main__":
    main()
