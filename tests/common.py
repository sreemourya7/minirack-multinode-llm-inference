"""Helpers shared by the per-part correctness checks."""
import os
import re
import signal
import socket
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

LENGTHS = [1, 2, 7, 16, 32]


def free_ports(n):
    """Asks the OS for n unused TCP ports."""
    socks = [socket.socket() for _ in range(n)]
    for s in socks:
        s.bind(("127.0.0.1", 0))
    ports = [s.getsockname()[1] for s in socks]
    for s in socks:
        s.close()
    return ports


def test_sequences():
    """Random token sequences of several lengths, plus one real sentence."""
    rng = np.random.default_rng(0)
    sequences = [rng.integers(0, 50257, size=n).tolist() for n in LENGTHS]
    sequences.append([15496, 11, 616, 3290, 318, 13779])  # "Hello, my dog is cute"
    return sequences


def reference_logits(hf_dir, sequences):
    """Logits from Hugging Face GPT-2 in PyTorch: the answer we must match."""
    import torch
    from transformers import GPT2LMHeadModel

    torch.set_grad_enabled(False)
    model = GPT2LMHeadModel.from_pretrained(hf_dir, local_files_only=True).eval()
    out = []
    for ids in sequences:
        logits = model(torch.tensor([ids], dtype=torch.long)).logits[0]
        out.append(logits.numpy().astype(np.float32))
    return out, model.config.vocab_size, model.config.n_positions


class TpRack:
    """Starts N tensor-parallel ranks (root first) and stops them on exit.

    moe: optional path to an MoE sidecar file, passed to every rank.
    env: extra environment variables for the rank processes.
    extra: extra command-line options for the root only.
    """

    def __init__(self, binary, weights, nranks, log_dir, tag="", env=None, moe=None, extra=None):
        self.procs = []
        self.spares = []
        self.logs = []
        self.cmds = []
        self.env = dict(os.environ, **(env or {}))
        (self.port,) = free_ports(1)
        for r in range(nranks):
            cmd = [binary, "--rank", str(r), "--nranks", str(nranks), "--weights", weights]
            cmd += ["--port", str(self.port)] if r == 0 else ["--root", f"127.0.0.1:{self.port}"]
            if moe:
                cmd += ["--moe", moe]
            if r == 0 and extra:
                cmd += list(extra)  # root-only options, e.g. --recovery checkpoint
            path = os.path.join(log_dir, f"tp{tag}_{nranks}ranks_rank{r}.log")
            open(path, "w").close()
            self.logs.append(path)
            self.cmds.append(cmd)
            self.procs.append(None)
            self.start(r)
        self._wait_ready()

    def start(self, r):
        """Starts (or, after a failure, replaces) rank r. Its log is appended to."""
        self.procs[r] = subprocess.Popen(self.cmds[r], stderr=open(self.logs[r], "a"), stdout=subprocess.DEVNULL,
                                         env=self.env)

    def start_standby(self, r):
        """Starts a spare process for rank r. The root keeps it waiting until rank r fails."""
        proc = subprocess.Popen(self.cmds[r], stderr=open(self.logs[r], "a"), stdout=subprocess.DEVNULL, env=self.env)
        self.spares.append(proc)
        return proc

    def kill(self, r, sig=signal.SIGKILL):
        """Sends rank r a signal. SIGKILL: it dies at once. SIGSTOP: it hangs."""
        self.procs[r].send_signal(sig)
        if sig == signal.SIGKILL:
            self.procs[r].wait()

    def lines(self, rank, word):
        """Parsed key=value fields of the log lines of `rank` that contain `word`."""
        rows = []
        with open(self.logs[rank]) as f:
            for line in f:
                if word in line:
                    row = {k: float(v) for k, v in re.findall(r"(\w+)=(-?[\d.]+)(?=\s|$)", line)}
                    row.update({k: v for k, v in re.findall(r"(\w+)=([a-z_]+)(?=\s|$)", line)})
                    rows.append(row)
        return rows

    def _wait_ready(self, timeout=120.0):
        from mr_client import Client

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

    def stats(self, rank=0):
        """Parses one rank's per-request log lines into dicts (oldest first)."""
        rows = []
        with open(self.logs[rank]) as f:
            for line in f:
                if " req " in line and "total_ms" in line:
                    row = {k: float(v) for k, v in re.findall(r"(\w+)=([\d.]+)", line)}
                    load = re.search(r"expert_load=\[([\d,]*)\]", line)
                    row["expert_load"] = [int(x) for x in load.group(1).split(",")] if load and load.group(1) else []
                    op = re.search(r"op=(\w+)", line)
                    row["op"] = op.group(1) if op else "forward"
                    rows.append(row)
        return rows

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        everyone = self.procs + self.spares
        for p in everyone:
            if p.poll() is None:
                p.send_signal(signal.SIGCONT)  # in case a test left it stopped
                p.terminate()
        for p in everyone:
            p.wait(timeout=10)


def moe_reference_logits(hf_dir, moe_path, sequences):
    """Logits from a PyTorch MoE model: Hugging Face GPT-2 with every block's
    MLP replaced by the MoE layer described in src/moe.h.

    The reference is written the opposite way round from the C++ code: it runs
    EVERY hidden unit for every token and then multiplies by a 0/gate mask,
    whereas the C++ runs only the chosen experts. Two different routes to the
    same numbers make a more convincing check.

    Returns (logits per sequence, expert load per sequence, smallest routing
    margin, (n_expert, top_k)).
    The margin is the smallest gap between the k-th and (k+1)-th router score
    over every routing decision; if it were tiny, float noise could flip a
    decision and the comparison would be meaningless.
    """
    import torch
    from transformers import GPT2LMHeadModel

    from mrkw import read_mrkw

    torch.set_grad_enabled(False)
    _, t = read_mrkw(moe_path)
    E, k, softmax = int(t["moe.config"][0, 0]), int(t["moe.config"][0, 1]), bool(t["moe.config"][0, 2])
    track = {"load": torch.zeros(E, dtype=torch.long), "margin": float("inf")}

    class MoEMLP(torch.nn.Module):
        def __init__(self, mlp, w, b):
            super().__init__()
            self.mlp, self.w, self.b = mlp, torch.from_numpy(w.copy()), torch.from_numpy(b.copy())

        def forward(self, x):
            scores = x @ self.w + self.b                               # [..., E]
            order = scores.sort(dim=-1, descending=True)
            vals, idx = order.values[..., :k], order.indices[..., :k]
            if k < E:
                track["margin"] = min(track["margin"], float((order.values[..., k - 1] - order.values[..., k]).min()))
            g = torch.softmax(vals, dim=-1) if softmax else torch.ones_like(vals)
            gates = torch.zeros_like(scores).scatter(-1, idx, g)       # [..., E], zero for unchosen experts
            track["load"] += torch.bincount(idx.flatten(), minlength=E)
            up = self.mlp.act(self.mlp.c_fc(x))                        # every hidden unit, [..., U]
            unit_gate = gates.repeat_interleave(up.shape[-1] // E, dim=-1)
            return (up * unit_gate) @ self.mlp.c_proj.weight + self.mlp.c_proj.bias

    model = GPT2LMHeadModel.from_pretrained(hf_dir, local_files_only=True).eval()
    for i, block in enumerate(model.transformer.h):
        block.mlp = MoEMLP(block.mlp, t[f"h.{i}.moe.router.weight"], t[f"h.{i}.moe.router.bias"])

    logits, loads = [], []
    for ids in sequences:
        track["load"].zero_()
        out = model(torch.tensor([ids], dtype=torch.long)).logits[0]
        logits.append(out.numpy().astype(np.float32))
        loads.append(track["load"].tolist())
    return logits, loads, track["margin"], (E, k)


def hf_greedy(hf_dir, prompt, n_new):
    import torch
    from transformers import GPT2LMHeadModel

    torch.set_grad_enabled(False)
    model = GPT2LMHeadModel.from_pretrained(hf_dir, local_files_only=True).eval()
    ids = torch.tensor([prompt], dtype=torch.long)
    out = model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=n_new, do_sample=False,
                         pad_token_id=50256)
    return out[0, len(prompt):].tolist()


def detokenize(hf_dir, ids):
    """Text for token ids, or None if the tokenizer files are not there."""
    try:
        from transformers import GPT2TokenizerFast

        return GPT2TokenizerFast.from_pretrained(hf_dir, local_files_only=True).decode(ids)
    except Exception:
        return None
