# Architecture

How minirack is put together: the processes, what they say to each other, how
one request moves through them, where state lives, and what happens when a
process dies. The reasons behind each choice are in the per-part
[notes](README.md); the measurements are in [RESULTS.md](RESULTS.md).

## 1. The pieces

```
                                 +--------------------+
  Python client  <--- TCP --->   |  rank 0 (root)     |  <--- TCP --->  rank 1
  scripts/mr_client.py           |  front door        |  <--- TCP --->  rank 2
                                 |  + its own shard   |  <--- TCP --->  rank 3
                                 +--------------------+
                                          ^
                                          |  a spare process for rank 1, already loaded,
                                          +- waiting to take over (a "standby")
```

| Piece | What it is |
|-------|------------|
| **Rank** | One `mr_rank` process. It stands in for one node of a rack and holds one shard of the model. |
| **Root** | Rank 0. A rank like the others, and also the only one clients talk to. |
| **Mesh** | The connections between ranks: a star, every rank holding one socket to the root. |
| **Client** | Anything that speaks the wire protocol. `scripts/mr_client.py` is a small Python one. |
| **Standby** | An extra process started with a rank number that is already in use. The root keeps it waiting. |
| **Supervisor** | `scripts/supervisor.py`. Starts the ranks and restarts any that exit. |

Part 1 uses a different, simpler layout: `mr_worker` processes in a chain,
each running a range of layers and passing the hidden state to the next.

## 2. How the model is cut

GPT-2 small has 12 layers. Each layer has 12 attention heads and an MLP with
3072 hidden units. The last step multiplies by a 50257-row vocabulary matrix.

With N ranks, rank r owns the r-th slice of each of those:

```
                   rank 0                 rank 1
 attention heads   0-5                    6-11
 MLP hidden units  0-1535                 1536-3071
 (or MoE experts)  0-3                    4-7
 vocabulary rows   0-25127                25128-50256
 small tensors     all of them (layer norms, biases)   <- copied on every rank
```

The weight file (`.mrkw`) has its index at the front, so a rank looks up a
tensor by name and reads only the rows it owns. No rank loads the whole model.

Every rank also keeps its own full copy of the **hidden state** `h`, the
`[tokens x 768]` matrix that flows through the layers. Those copies have to
stay identical, and most of the design follows from that.

## 3. The wire protocol

TCP delivers a stream of bytes with no boundaries, so every message starts
with a fixed 32-byte header that says what follows.

| Offset | Size | Field | Meaning |
|-------:|-----:|-------|---------|
| 0 | 4 | magic | `MRK1`. Catches garbage and wrong peers. |
| 4 | 4 | type | See the table below. |
| 8 | 4 | flags | `LAST_ONLY`, `ARGMAX`, `BACKUP`. |
| 12 | 4 | seq | Which step of the request this message belongs to. |
| 16 | 8 | request_id | Echoed in the reply. |
| 24 | 8 | payload_len | Bytes of payload that follow. |

All numbers are little-endian. Matrices travel as `u32 rows, u32 cols`, then
raw float32.

| Type | Direction | Payload |
|------|-----------|---------|
| `TOKENS` | client to root | Token ids: run the whole sequence, keep nothing |
| `PREFILL` | client to root | Session id and the prompt's token ids |
| `DECODE` | client to root | Session id and one token |
| `RELEASE` | client to root | Session id |
| `PING` / `PONG` | client and root | Is the rack ready? |
| `LOGITS` | root to client | A matrix of logits |
| `TOKEN` | root to client | The chosen next token and its logit (with `ARGMAX`) |
| `OK`, `ERROR` | root to client | Nothing, or a text message |
| `HELLO` | rank to root | Rank number, rack size, and which kernels it runs |
| `START` | root to every rank | What to do, and the embedded input |
| `COLL` | between ranks | Data inside a collective |
| `ABORT` / `ABORT_ACK` | root and rank | Drop the request in progress; confirmed |

`TCP_NODELAY` is set on every socket. A decode step sends many small messages,
and the default batching of small packets would add a delay to each one.

## 4. One request, end to end

Every rank runs **the same program** (`execute()` in `src/exec.cpp`) on its
own slice of the weights. Most lines are local work. A few are
**collectives**: points where all ranks stop, exchange data, and carry on with
a shared result.

| Collective | What it does | Used for |
|------------|--------------|----------|
| `start_request` | The root gives every rank the same request and input | Start of every request |
| `all_reduce_sum` | Every rank puts in a matrix; every rank gets back the sum | Twice per layer |
| `check_identical` | Every rank sends a checksum of `h`; the root compares them | Once, after the last layer |
| `gather_cols` | Every rank sends some columns; the root joins them | The logits, or each rank's best token |
| `gather_parts` | Every rank sends a matrix; the root keeps them apart | Checkpoints of the cache |

One layer on two ranks:

```
            rank 0                                        rank 1
            ------                                        ------
 h  (same on both ranks)
            layer norm                                    layer norm
            q, k, v for heads 0-5                         q, k, v for heads 6-11
            attention for those heads                     attention for those heads
            project: its share of the output              project: its share of the output
                          \                                /
                           +------ all_reduce_sum --------+        sum over all 12 heads
            add bias, add to h                            add bias, add to h

            layer norm                                    layer norm
            hidden units 0-1535, GELU                     hidden units 1536-3071, GELU
            project: its share of the output              project: its share of the output
                          \                                /
                           +------ all_reduce_sum --------+        sum over all 3072 units
            add bias, add to h                            add bias, add to h
 h  (same on both ranks again)
```

The whole request, for one decode step:

```
 client          root (rank 0)                              rank 1
   |  DECODE       |                                           |
   |-------------->|  check the request, embed the token       |
   |               |---- START (request, input) -------------->|
   |               |                                           |
   |               |<======= 12 layers, 2 all-reduces each ===>|
   |               |<------- checksum of h ---------------------|   check_identical
   |               |<------- best token in its vocabulary ------|   gather_cols
   |               |  commit: record the token (section 6)     |
   |  TOKEN        |                                           |
   |<--------------|                                           |
```

That is 27 exchanges per decode step: 1 start, 24 all-reduces, 1 check and
1 gather.

**How the all-reduce works on a star.** Every rank sends its matrix to the
root. The root adds them, always in the order rank 0 + rank 1 + rank 2 + ...,
and sends the sum back to everyone. Floating-point addition gives slightly
different answers in different orders, so forming the sum in one place and
one order is what makes every rank's copy of `h` identical to the last bit.

**Why the bias is added after the all-reduce.** A bias belongs to the whole
layer, not to one slice. If every rank added it before the sum, it would be
counted once per rank.

**Sequence numbers.** Each collective in a request has a number. A rank that
receives a message with the wrong number has fallen out of step, and fails
loudly instead of computing with the wrong data.

**The vocabulary step.** With `ARGMAX`, each rank finds the best token in its
own slice of the vocabulary and sends 8 bytes (logit and token id). The root
keeps the best. Without it, each rank sends its columns of logits and the
root joins them.

### Mixture of Experts

With `--moe`, the MLP half of each layer changes. The 3072 hidden units are
grouped into 8 experts of 384. A small router scores the experts for each
token, the best k (2 by default) are kept, and only those run.

```
 x (same on every rank)
   |
   router: 8 scores per token --> keep the top 2 --> gates = softmax of those 2
   |                                                   (every rank computes this itself:
   |                                                    same input, same code, same answer)
   +--> rank 0: for each of experts 0-3, gather the tokens that chose it, run it, scatter back
   +--> rank 1: the same for experts 4-7
   |
   all_reduce_sum, then one shared bias
```

Routing needs no messages, because every rank holds the same `x`. This is
where bit-identical copies stop being a nicety: the router makes hard choices,
so a difference in the last bit of `x` could send a token to a different
expert on one rank than on another.

## 5. State: sessions and the KV cache

A **session** is the stored state of one piece of text being generated. For
every layer it holds the key and value of every token so far.

```
 PREFILL  run the whole prompt once, store every token's key and value
 DECODE   run one new token: append its key and value, attend over all stored
 RELEASE  free the session on every rank
```

Each rank stores keys and values only for the heads it owns. The cache is cut
exactly like the weights and never crosses the wire in normal operation.

| Rule | Why |
|------|-----|
| Decode with the cache gives the same bits as recomputing everything | Tested directly. Recovery by replay depends on it. |
| The root sends the length it expects with every decode; each rank checks its own | A cheap test that every rank has seen the same history. |
| At most 64 sessions | A bound on memory. Over the limit, the request is refused before any rank hears of it. |
| A client that disconnects has its sessions released | A crashed client cannot leak cache memory. |
| Every request is checked on the root before `START` | A bad request never leaves the ranks half-way through a step. |

## 6. Failure and recovery

The failure handled: a non-root rank dies or freezes at any moment, including
in the middle of a step. The root itself is not protected.

**One rule makes recovery possible: a request has only happened once it has
finished on every rank.** Until then the root records nothing. So after any
failure there is a well-defined last point at which every rank agreed, and
the rack can be returned to it.

```
 1. NOTICE     a rank's socket closes (crash), or it stays silent past
               --rank-timeout-ms (hang)
 2. FENCE      the root closes that socket. A rank that was only frozen finds
               its connection gone when it wakes, and exits. It can never
               rejoin with stale state.
 3. ABORT      the root tells each surviving rank to drop the request, and
               waits for each to confirm, so no stale message is left in a socket
 4. REPLACE    a waiting standby takes over at once; otherwise the root waits
               up to --rejoin-timeout-ms for a new process to say HELLO
 5. ROLL BACK  the root cuts its own cache back to the last committed token;
               every other rank drops its sessions
 6. REBUILD    replay:     run each session's tokens again as one prefill
               checkpoint: hand each rank back the cache rows it had sent
 7. RETRY      the interrupted request runs again; the client gets its answer
```

The two ways to rebuild:

| | Replay | Checkpoint |
|---|--------|------------|
| Cost while healthy | None | Each rank sends its new cache rows to the root after every step |
| Extra memory | The token ids | A second copy of every other rank's cache, on the root |
| Rebuild time | Grows with the length of the text | A copy; small and nearly flat |

The root's event loop watches everything with one `poll()`: new connections,
the current client, and the sockets of the other ranks. A rank's socket
should be silent between requests, so if it becomes readable the rank has
gone, and the root learns that at once, before the next request arrives. A
new connection is not read from until it has sent something, so a connection
that stays silent cannot hold the root up.

A replacement's `HELLO` states which kernels it runs. The scalar and AVX2
kernels differ in the last bit, so a rank with different kernels is refused:
mixing them would break the identical-copies rule.

## 7. Kernels and profiling

All arithmetic goes through four routines in `src/kernels.*`:

| Routine | Used for |
|---------|----------|
| `gemm_acc` | Matrix multiply: every linear layer and projection |
| `rows_dot` | The vocabulary step: one vector against many rows |
| `dot` | Attention scores |
| `axpy` | Attention output: add a scaled row |

Each has a plain C++ version and an AVX2 + FMA version. The choice is made
once at start-up: AVX2 if the CPU has it, unless `MR_KERNELS=scalar` is set.
Only `kernels_avx2.cpp` is compiled with `-mavx2 -mfma`, so the binary still
runs on a CPU without them.

The AVX2 matrix multiply produces the same bits for a token whether it is
computed alone or in a batch. Without that property, decode (one token) and
prefill (many tokens) would disagree, and the cache would no longer be exact.

Timers in `src/prof.*` split each request into matrix multiply, attention,
GELU, layer norm, the vocabulary step and communication. Every request writes
one `key=value` line to stderr; the tests and the chart scripts read those.

## 8. How it is tested

| Technique | Example |
|-----------|---------|
| A reference implementation | Logits compared with Hugging Face GPT-2 in PyTorch |
| Exact equality where it should hold | Any number of ranks gives identical bits; cached decode equals uncached |
| An anchor case | MoE with all 8 experts on and gates of 1 must equal plain GPT-2 |
| A conservation check | The tokens the experts processed must add up to tokens x layers x k |
| Proving a check can fail | One rank is made to corrupt `h`; the replica check must catch it |
| Forced failures | `kill -9` and `SIGSTOP` at chosen moments, then compare with an undisturbed run |
| A second implementation as the oracle | The AVX2 kernels are tested against the scalar ones |

`scripts/run_all_checks.sh` builds everything and runs all six check scripts.

## 9. Source map

| File | Role |
|------|------|
| `src/net.*` | Sockets, framing, timeouts |
| `src/mesh.*` | Collectives, membership, abort and rejoin |
| `src/weights.*` | The `.mrkw` reader: whole tensors or row slices |
| `src/tp_model.*` | A rank's shard, and the layer loop |
| `src/moe.*` | Router, top-k, gates, running the chosen experts |
| `src/kv_cache.*` | Sessions, the cache, attention that uses it |
| `src/exec.*` | The program every rank runs for one request |
| `src/root.*` | The root: front door, event loop, recovery |
| `src/rank.cpp` | `main()`, options, the loop of the other ranks |
| `src/kernels.*`, `src/kernels_avx2.cpp` | The hot loops |
| `src/prof.*` | Per-operation timers |
| `src/ops.*`, `src/model.*`, `src/worker.cpp` | Part 1: single-process operations and the pipeline worker |
| `src/bench.cpp`, `src/kernel_test.cpp` | Microbenchmark and kernel unit tests |
