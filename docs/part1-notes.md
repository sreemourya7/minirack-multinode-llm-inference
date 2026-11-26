# Part 1 notes: concepts and decisions

Part 1 answers one question: **can a model be cut into pieces that live in
different processes and still give the right answer?** Everything later
(collectives, KV cache, failure recovery, SIMD) sits on top of this.

```
client --tokens--> worker 0 ----hidden [T x 768]----> worker 1
                   embed + layers 0-5                 layers 6-11 + head
client <--logits-- worker 0 <-------logits----------- worker 1
```

## Concepts

**1. Shard.** The slice of the model one worker owns: a range of layers, plus
the embedding if it is first, plus the head if it is last.
Code: `Shard` in `src/model.h`.

**2. Pipeline (layer-wise) split.** Cut the model between two layers. The only
thing that has to cross the cut is the hidden state, a `[T, 768]` matrix.
Example: 32 tokens is 32 x 768 x 4 bytes = 98 KB per hop. Small, compared with
the 500 MB of weights that stay put.

**3. The GPT-2 block.** Two halves, each wrapped in a residual connection:
`h = h + proj(attention(ln1(h)))`, then `h = h + mlp(ln2(h))`. The layernorm
comes *before* each half ("pre-LN").
Code: `run_blocks` in `src/model.cpp`.

**4. Causal attention.** Token i scores itself against tokens 0..i only, turns
the scores into probabilities with softmax, and takes a weighted sum of their
values. The "mask" is just the loop bound `j <= i`.
Code: `causal_self_attention` in `src/ops.cpp`.

**5. Weight tying.** GPT-2 uses one matrix (`wte`, 154 MB) both to turn tokens
into vectors and to turn vectors back into token scores. So the first and the
last worker each load a copy. That is why a worker holding one layer plus the
embedding is still 186 MB.

**6. Message framing.** TCP delivers a stream of bytes with no boundaries. So
each message starts with a fixed 32-byte header saying what it is and how many
bytes follow. Read the header, then read exactly that many bytes.
Code: `send_msg` / `recv_msg` in `src/net.cpp`.

**7. Partial reads and writes.** `send()` and `recv()` may move fewer bytes
than asked. Every real network program wraps them in a loop.
Code: `send_all` / `recv_all`.

**8. Request ID.** The client stamps each request with a number and every
reply echoes it. Today it is a sanity check. Later it is what lets several
requests share one connection and what state (the KV cache) hangs off.

**9. Golden-reference testing.** A separate, trusted implementation (Hugging
Face GPT-2 in PyTorch) says what the answer should be. The C++ output must
match to within floating-point noise.

**10. Why the match is "close", not "exact".** Float addition is not
associative: `(a + b) + c` can differ from `a + (b + c)` in the last bit.
PyTorch's matmul adds in a different order from our loops, so logits differ by
about 3e-4 on values of size about 100. That is a relative error near 1e-6,
which is float32 precision. But two runs of *our own* code add in the same
order, so 1 worker and 3 workers agree bit for bit. The test checks both.

## Decisions

| Decision | Why | What was given up |
|----------|-----|-------------------|
| Split by layers first, not inside a layer | Needs no collectives; the simplest thing that is truly multi-node | No speedup for a single request: workers take turns |
| Own weight format with the index at the front | Each worker reads only its shard; no JSON parser in C++ | One extra conversion step |
| Raw TCP and a hand-written binary protocol, not gRPC or MPI | The communication layer is the subject of the project; zero dependencies | Features those libraries give for free |
| Blocking sockets, one request at a time | Easy to read and to reason about | No concurrency or batching yet |
| Replies travel back along the chain | One connection per hop; the client only knows worker 0 | Logits are copied once per hop (use `last_only` to keep that small) |
| `TCP_NODELAY` on every socket | Request/reply traffic should not wait for Nagle's batching | Nothing for this traffic pattern |
| A bad request gets an error reply; the worker stays up | One bad input must not take a node down | A little extra code in the request loop |
| Plain scalar loops, `-O2`, no `-march=native` | An honest baseline for Part 5 to beat | Speed, for now |
| No KV cache yet | Correctness first | Every request recomputes all tokens, so time grows with length |
| Linux / POSIX sockets | What inference infrastructure runs on | Needs WSL2 on Windows |
| Hugging Face as the reference, not our own PyTorch code | An independent implementation catches mistakes a self-written one would repeat | A heavier test dependency |

## Questions worth being able to answer

- What crosses the wire between two workers, and how big is it?
- Why does splitting one request across workers not make it faster? When does
  it help?
- Why must `recv` be called in a loop?
- Why is the tolerance against PyTorch 2e-3 and not zero, while the tolerance
  between splits is exactly zero?
- Which worker needs `wte`, and why two of them?
