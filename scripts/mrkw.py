"""Reader and writer for minirack's .mrkw weight files (layout in src/weights.h)."""
import struct

import numpy as np

MAGIC = b"MRKW"
VERSION = 1
CONFIG_KEYS = ("n_layer", "n_head", "n_embd", "vocab_size", "n_ctx")


def write_mrkw(path, cfg, tensors):
    """cfg: dict with CONFIG_KEYS. tensors: list of (name, 2-D float array)."""
    tensors = [(name, np.ascontiguousarray(arr, dtype="<f4")) for name, arr in tensors]
    for name, arr in tensors:
        if arr.ndim != 2:
            raise ValueError(f"{name}: expected a 2-D array, got shape {arr.shape}")

    # Work out where each tensor's data will sit, so the index can go first.
    index_size = sum(4 + len(name.encode()) + 4 + 4 + 8 for name, _ in tensors)
    offset = 4 + 4 + 5 * 4 + 4 + index_size
    offsets = []
    for _, arr in tensors:
        offsets.append(offset)
        offset += arr.size * 4

    with open(path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<I", VERSION))
        f.write(struct.pack("<5I", *(cfg[k] for k in CONFIG_KEYS)))
        f.write(struct.pack("<I", len(tensors)))
        for (name, arr), off in zip(tensors, offsets):
            raw = name.encode()
            f.write(struct.pack("<I", len(raw)))
            f.write(raw)
            f.write(struct.pack("<IIQ", arr.shape[0], arr.shape[1], off))
        for (name, arr), off in zip(tensors, offsets):
            assert f.tell() == off, name
            f.write(arr.tobytes())
    return offset  # total file size in bytes


def read_mrkw(path, load_tensors=True):
    """Returns (cfg dict, {name: float32 array}). With load_tensors=False the
    dict is empty and only the header is read (cheap for a big file)."""
    with open(path, "rb") as f:
        if f.read(4) != MAGIC:
            raise ValueError(f"{path} is not a .mrkw file")
        (version,) = struct.unpack("<I", f.read(4))
        if version != VERSION:
            raise ValueError(f"unsupported .mrkw version {version}")
        cfg = dict(zip(CONFIG_KEYS, struct.unpack("<5I", f.read(20))))
        (n,) = struct.unpack("<I", f.read(4))
        index = []
        for _ in range(n):
            (name_len,) = struct.unpack("<I", f.read(4))
            name = f.read(name_len).decode()
            rows, cols, off = struct.unpack("<IIQ", f.read(16))
            index.append((name, rows, cols, off))
        tensors = {}
        if load_tensors:
            for name, rows, cols, off in index:
                f.seek(off)
                tensors[name] = np.frombuffer(f.read(rows * cols * 4), dtype="<f4").reshape(rows, cols)
    return cfg, tensors
