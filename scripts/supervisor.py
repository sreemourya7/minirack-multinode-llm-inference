#!/usr/bin/env python3
"""Keeps a rack running: starts every rank and replaces any that exits.

    python3 scripts/supervisor.py --weights weights/gpt2.mrkw --nranks 2
    python3 scripts/supervisor.py --weights weights/gpt2.mrkw --nranks 2 --standby --recovery checkpoint

The root (rank 0) repairs the rack once a replacement connects, but something
has to START that replacement. That is this script's job. In a real cluster
the same role is played by the cluster's process manager.

--standby keeps one spare process per rank already running with its weights
loaded. When a rank fails, the root switches to the spare at once, and this
script starts a new spare in the background.

Try it: start a rack, run scripts/generate.py in another terminal, and
`kill -9` a rank while it is generating.
"""
import argparse
import os
import signal
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from mr_client import Client  # noqa: E402


class Supervisor:
    def __init__(self, binary, weights, nranks, port=9100, moe=None, root_args=(), standby=False, log_dir=None,
                 on_event=None):
        self.nranks, self.port, self.standby = nranks, port, standby
        self.on_event = on_event or (lambda text: print(f"[supervisor] {text}", file=sys.stderr, flush=True))
        self.log_dir = log_dir
        self.cmds = []
        for r in range(nranks):
            cmd = [binary, "--rank", str(r), "--nranks", str(nranks), "--weights", weights]
            cmd += ["--port", str(port)] + list(root_args) if r == 0 else ["--root", f"127.0.0.1:{port}"]
            if moe:
                cmd += ["--moe", moe]
            self.cmds.append(cmd)
        self.procs = {}   # rank -> the process believed to be serving that rank
        self.spares = {}  # rank -> its standby process, if any
        self.stopping = False
        self.lock = threading.Lock()
        self.thread = None

    def _spawn(self, r):
        if self.log_dir:
            os.makedirs(self.log_dir, exist_ok=True)
            err = open(os.path.join(self.log_dir, f"rank{r}.log"), "a")
        else:
            err = subprocess.DEVNULL
        return subprocess.Popen(self.cmds[r], stderr=err, stdout=subprocess.DEVNULL)

    def start(self, timeout=120.0):
        for r in range(self.nranks):
            self.procs[r] = self._spawn(r)
        deadline = time.time() + timeout
        while True:  # wait until the root answers a ping: every rank has joined
            if time.time() > deadline:
                raise RuntimeError("the rack did not become ready")
            try:
                with Client("127.0.0.1", self.port, timeout=5.0) as c:
                    if c.ping():
                        break
            except OSError:
                time.sleep(0.2)
        if self.standby:
            for r in range(1, self.nranks):
                self.spares[r] = self._spawn(r)
        self.thread = threading.Thread(target=self._watch, daemon=True)
        self.thread.start()
        return self

    def _watch(self):
        while not self.stopping:
            time.sleep(0.02)
            with self.lock:
                if self.procs[0].poll() is not None and not self.stopping:
                    self.on_event("the root exited; a rack cannot survive that (see the README), stopping")
                    self.stopping = True
                    break
                for r in range(1, self.nranks):
                    spare = self.spares.get(r)
                    if spare is not None and spare.poll() is not None:
                        self.on_event(f"the standby for rank {r} exited; starting another")
                        self.spares[r] = self._spawn(r)
                    if self.procs[r].poll() is None:
                        continue
                    code = self.procs[r].returncode
                    how = f"killed by signal {-code}" if code < 0 else f"exit code {code}"
                    if r in self.spares:
                        # The root has switched (or will switch) to the spare. It
                        # becomes the serving process, and we line up a new spare.
                        self.on_event(f"rank {r} is gone ({how}); its standby takes over, starting a new standby")
                        self.procs[r] = self.spares[r]
                        self.spares[r] = self._spawn(r)
                    else:
                        self.on_event(f"rank {r} is gone ({how}); starting a replacement")
                        self.procs[r] = self._spawn(r)

    def kill(self, r, sig=signal.SIGKILL):
        """For demos and tests: make rank r fail."""
        with self.lock:
            self.procs[r].send_signal(sig)

    def stop(self):
        self.stopping = True
        if self.thread:
            self.thread.join(timeout=2)
        everyone = list(self.procs.values()) + list(self.spares.values())
        for p in everyone:
            if p.poll() is None:
                p.send_signal(signal.SIGCONT)
                p.terminate()
        for p in everyone:
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                p.kill()

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--nranks", type=int, default=2)
    ap.add_argument("--port", type=int, default=9100)
    ap.add_argument("--moe")
    ap.add_argument("--recovery", choices=["replay", "checkpoint"], default="replay")
    ap.add_argument("--standby", action="store_true", help="keep a spare process per rank ready to take over")
    ap.add_argument("--rank-timeout-ms", type=int, default=10000)
    ap.add_argument("--log-dir", default=os.path.join(HERE, "..", "build", "logs", "supervised"))
    ap.add_argument("--rank-binary", default=os.path.join(HERE, "..", "build", "mr_rank"))
    args = ap.parse_args()

    root_args = ["--recovery", args.recovery, "--rank-timeout-ms", str(args.rank_timeout_ms)]
    sup = Supervisor(args.rank_binary, args.weights, args.nranks, args.port, args.moe, root_args, args.standby,
                     args.log_dir)
    sup.start()
    pids = ", ".join(f"rank {r}: pid {p.pid}" for r, p in sup.procs.items())
    print(f"[supervisor] rack is up on port {args.port} ({pids}). Ctrl-C to stop.", file=sys.stderr, flush=True)
    try:
        while not sup.stopping:
            time.sleep(0.2)
    except KeyboardInterrupt:
        pass
    finally:
        sup.stop()


if __name__ == "__main__":
    main()
