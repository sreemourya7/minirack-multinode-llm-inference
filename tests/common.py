"""Helpers shared by the per-part correctness checks."""
import os
import socket
import sys

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
