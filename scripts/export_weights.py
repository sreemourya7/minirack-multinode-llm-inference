#!/usr/bin/env python3
"""Convert GPT-2's Hugging Face weight file into minirack's .mrkw format.

    python3 scripts/export_weights.py --safetensors model.safetensors \
        --config config.json --out gpt2.mrkw

Needs only numpy. The .safetensors format is simple enough to read by hand:
8 bytes giving the length of a JSON header, the JSON header (name -> dtype,
shape, byte range), then the raw tensor bytes.

Why convert at all? So the C++ side needs no JSON parser, and so the index is
at the front of the file and a worker can read just its own shard.
"""
import argparse
import json
import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mrkw import CONFIG_KEYS, write_mrkw  # noqa: E402


def read_safetensors(path):
    """Returns {name: float32 array}, memory-mapped so nothing is copied yet."""
    with open(path, "rb") as f:
        (header_len,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(header_len))
    data_start = 8 + header_len
    blob = np.memmap(path, dtype=np.uint8, mode="r", offset=data_start)
    tensors = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        if info["dtype"] != "F32":
            raise ValueError(f"{name}: expected F32, got {info['dtype']}")
        begin, end = info["data_offsets"]
        arr = np.frombuffer(blob[begin:end], dtype="<f4").reshape(info["shape"])
        tensors[name] = arr
    return tensors


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--safetensors", required=True)
    ap.add_argument("--config", required=True, help="the model's config.json")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)
    tensors = read_safetensors(args.safetensors)

    # Keep 1-D and 2-D float tensors. "h.N.attn.bias" is not a weight: it is
    # the causal mask Hugging Face stores as a buffer, and we build the mask in
    # code instead. 1-D tensors (biases, layernorm) are stored as one row.
    keep = []
    for name in sorted(tensors):
        arr = tensors[name]
        if name.endswith(".attn.bias") or name.endswith(".attn.masked_bias"):
            continue
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.ndim != 2:
            raise ValueError(f"{name}: unexpected shape {arr.shape}")
        keep.append((name, arr))

    size = write_mrkw(args.out, {k: cfg[k] for k in CONFIG_KEYS}, keep)
    print(f"wrote {args.out}: {len(keep)} tensors, {size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
