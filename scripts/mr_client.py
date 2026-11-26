"""Python client for minirack. Speaks the wire protocol described in src/net.h.

    from mr_client import Client
    with Client("127.0.0.1", 9000) as c:
        logits = c.forward([15496, 11, 995])       # [T, vocab] numpy array
        last = c.forward([15496, 11, 995], last_only=True)   # [1, vocab]
"""
import socket
import struct

import numpy as np

MAGIC = 0x314B524D  # "MRK1"
HEADER = struct.Struct("<IIIIQQ")  # magic, type, flags, reserved, request_id, payload_len

MSG_TOKENS = 1
MSG_HIDDEN = 2
MSG_LOGITS = 3
MSG_ERROR = 4

FLAG_LAST_ONLY = 1


class WorkerError(RuntimeError):
    """The rack answered with an error message instead of logits."""


class Client:
    def __init__(self, host, port, timeout=120.0):
        self.sock = socket.create_connection((host, port), timeout=timeout)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.next_id = 1

    def close(self):
        self.sock.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _recv_exact(self, n):
        buf = bytearray(n)
        view = memoryview(buf)
        got = 0
        while got < n:
            k = self.sock.recv_into(view[got:], n - got)
            if k == 0:
                raise ConnectionError("worker closed the connection")
            got += k
        return bytes(buf)

    def forward(self, token_ids, last_only=False):
        """Runs the whole chain on token_ids. Returns float32 logits [T or 1, vocab]."""
        ids = np.asarray(token_ids, dtype="<i4")
        payload = struct.pack("<I", ids.size) + ids.tobytes()
        req_id = self.next_id
        self.next_id += 1
        flags = FLAG_LAST_ONLY if last_only else 0
        self.sock.sendall(HEADER.pack(MAGIC, MSG_TOKENS, flags, 0, req_id, len(payload)) + payload)

        magic, mtype, _flags, _res, got_id, plen = HEADER.unpack(self._recv_exact(HEADER.size))
        if magic != MAGIC:
            raise ConnectionError("bad magic in reply")
        body = self._recv_exact(plen)
        if got_id != req_id:
            raise ConnectionError("reply is for a different request")
        if mtype == MSG_ERROR:
            raise WorkerError(body.decode("utf-8", "replace"))
        if mtype != MSG_LOGITS:
            raise ConnectionError(f"unexpected reply type {mtype}")
        rows, cols = struct.unpack_from("<II", body, 0)
        return np.frombuffer(body, dtype="<f4", offset=8).reshape(rows, cols)
