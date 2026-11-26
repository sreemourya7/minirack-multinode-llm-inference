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
import struct

import numpy as np

MAGIC = b"MRKW"
VERSION = 1


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

    # Work out where each tensor's data will sit, so the index can be written first.
    index_size = sum(4 + len(name.encode()) + 4 + 4 + 8 for name, _ in keep)
    header_size = 4 + 4 + 5 * 4 + 4
    offset = header_size + index_size
    offsets = []
    for _, arr in keep:
        offsets.append(offset)
        offset += arr.size * 4

    with open(args.out, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<I", VERSION))
        f.write(struct.pack("<5I", cfg["n_layer"], cfg["n_head"], cfg["n_embd"], cfg["vocab_size"], cfg["n_ctx"]))
        f.write(struct.pack("<I", len(keep)))
        for (name, arr), off in zip(keep, offsets):
            raw = name.encode()
            f.write(struct.pack("<I", len(raw)))
            f.write(raw)
            f.write(struct.pack("<IIQ", arr.shape[0], arr.shape[1], off))
        for (name, arr), off in zip(keep, offsets):
            assert f.tell() == off, name
            f.write(np.ascontiguousarray(arr, dtype="<f4").tobytes())

    print(f"wrote {args.out}: {len(keep)} tensors, {offset / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
