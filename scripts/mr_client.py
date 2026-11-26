"""Python client for minirack. Speaks the wire protocol described in src/net.h.

    from mr_client import Client
    with Client("127.0.0.1", 9100) as c:
        # Stateless: run a whole sequence, nothing is kept (Parts 1-3).
        logits = c.forward([15496, 11, 995])                  # [T, vocab]
        last = c.forward([15496, 11, 995], last_only=True)    # [1, vocab]

        # With a KV cache (Part 4, mr_rank only).
        new_tokens = c.generate([15496, 11, 995], n_new=20)   # greedy

        # ... or step by step:
        logits = c.prefill(session=1, token_ids=[15496, 11, 995])   # [vocab]
        logits = c.decode(session=1, token=int(logits.argmax()))    # [vocab]
        c.release(session=1)
"""
import socket
import struct

import numpy as np

MAGIC = 0x314B524D  # "MRK1"
HEADER = struct.Struct("<IIIIQQ")  # magic, type, flags, seq, request_id, payload_len

MSG_TOKENS = 1
MSG_HIDDEN = 2
MSG_LOGITS = 3
MSG_ERROR = 4
MSG_PING = 8
MSG_PONG = 9
MSG_PREFILL = 10
MSG_DECODE = 11
MSG_RELEASE = 12
MSG_TOKEN = 13
MSG_OK = 14

FLAG_LAST_ONLY = 1
FLAG_ARGMAX = 2


class WorkerError(RuntimeError):
    """The rack answered with an error message instead of a result."""


class Client:
    def __init__(self, host, port, timeout=120.0):
        self.sock = socket.create_connection((host, port), timeout=timeout)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.next_id = 1
        self.next_session = 1

    def close(self):
        self.sock.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # ------------------------------------------------------------- plumbing
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

    def _call(self, mtype, payload=b"", flags=0):
        """Sends one message and returns (reply type, reply payload)."""
        req_id = self.next_id
        self.next_id += 1
        self.sock.sendall(HEADER.pack(MAGIC, mtype, flags, 0, req_id, len(payload)) + payload)
        magic, rtype, _flags, _seq, got_id, plen = HEADER.unpack(self._recv_exact(HEADER.size))
        if magic != MAGIC:
            raise ConnectionError("bad magic in reply")
        body = self._recv_exact(plen)
        if got_id != req_id:
            raise ConnectionError("reply is for a different request")
        if rtype == MSG_ERROR:
            raise WorkerError(body.decode("utf-8", "replace"))
        return rtype, body

    @staticmethod
    def _logits(rtype, body):
        if rtype != MSG_LOGITS:
            raise ConnectionError(f"unexpected reply type {rtype}")
        rows, cols = struct.unpack_from("<II", body, 0)
        return np.frombuffer(body, dtype="<f4", offset=8).reshape(rows, cols)

    @staticmethod
    def _token(rtype, body):
        if rtype != MSG_TOKEN:
            raise ConnectionError(f"unexpected reply type {rtype}")
        return struct.unpack("<if", body)  # (token id, its logit)

    @staticmethod
    def _tokens_payload(token_ids):
        ids = np.asarray(token_ids, dtype="<i4")
        return struct.pack("<I", ids.size) + ids.tobytes()

    # ------------------------------------------------------------ stateless
    def ping(self):
        """True once an mr_rank rack is fully connected and ready for requests."""
        rtype, _ = self._call(MSG_PING)
        return rtype == MSG_PONG

    def forward(self, token_ids, last_only=False):
        """Runs a whole sequence with no cache. Returns float32 logits [T or 1, vocab]."""
        flags = FLAG_LAST_ONLY if last_only else 0
        return self._logits(*self._call(MSG_TOKENS, self._tokens_payload(token_ids), flags))

    # -------------------------------------------------- sessions (KV cache)
    def prefill(self, session, token_ids, argmax=False):
        """Opens a session and fills its KV cache from a prompt.

        Returns the logits for the next token, [vocab]. With argmax=True the
        rack picks the best token itself and this returns (token, logit).
        """
        payload = struct.pack("<Q", session) + self._tokens_payload(token_ids)
        reply = self._call(MSG_PREFILL, payload, FLAG_ARGMAX if argmax else 0)
        return self._token(*reply) if argmax else self._logits(*reply)[0]

    def decode(self, session, token, argmax=False):
        """Feeds one more token to a session. Same return value as prefill."""
        reply = self._call(MSG_DECODE, struct.pack("<Qi", session, token), FLAG_ARGMAX if argmax else 0)
        return self._token(*reply) if argmax else self._logits(*reply)[0]

    def release(self, session):
        """Frees a session's KV cache on every rank."""
        self._call(MSG_RELEASE, struct.pack("<Q", session))

    def generate(self, prompt_ids, n_new):
        """Greedy generation: one prefill, then n_new - 1 decode steps.
        The next token is chosen on the rack, so no logits cross the wire."""
        session = self.next_session
        self.next_session += 1
        out = []
        token, _ = self.prefill(session, prompt_ids, argmax=True)
        try:
            out.append(token)
            while len(out) < n_new:
                token, _ = self.decode(session, token, argmax=True)
                out.append(token)
        finally:
            self.release(session)
        return out
