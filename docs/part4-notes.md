# Part 4 notes: concepts and decisions

Parts 1 to 3 answered one question per request: "here is a sequence, what
comes next?" Generating text asks that question again and again, each time
with one more token. Part 4 answers: **how do we avoid redoing the old work
every time?** With per-request state: a KV cache.

```
 prefill "Hello, my dog is"  --> run 4 tokens once, store their keys and values (a session)
 decode  " cute"             --> run 1 token: append its key and value, attend over the 5 stored
 decode  "."                 --> run 1 token: append, attend over the 6 stored
 release                     --> free the session on every rank
```

## Concepts

**1. Autoregressive generation.** The model predicts one token, that token is
appended to the input, and the model runs again. Thirty new tokens means
thirty runs.

**2. Why old work can be reused.** Attention is causal: token 5 only looks at
tokens 0 to 5. So nothing that arrives later can change what was computed
for token 5. Its results are final the moment they are computed.

**3. KV cache.** In each layer, a token is seen by later tokens only through
its *key* and its *value*. So those are the two things worth keeping. The
cache stores one key and one value per token, per layer.
Code: `KvLayer` and `Session` in `src/kv_cache.h`.

**4. Prefill.** Run the whole prompt once, as before, and store every token's
key and value along the way. Many tokens at once.

**5. Decode.** Run one new token. In each layer: compute its query, key and
value, append the key and value to the cache, and let the query attend over
everything stored. One token of work, no matter how long the text is.
Code: `attend_cached` in `src/kv_cache.cpp`.

**6. Session.** The stored state of one conversation: its length, and the
cache. Identified by a number the client chooses. Created by prefill, extended
by decode, destroyed by release.

**7. Sharded state.** Each rank stores keys and values only for its own
attention heads, just as it stores only its own slice of the weights. The
cache never crosses the wire.

**8. Stateless versus stateful.** Until now a request left nothing behind, so
any failure could be fixed by sending it again. A session is state that lives
on every rank and must stay in step on all of them. That is new risk, and it
is what Part 5 has to deal with when a rank dies.

**9. Position check.** With each decode the root sends the session length it
expects. Every rank compares it with its own and stops if they differ. A
cheap test that all ranks' caches have seen the same history.

**10. Cached equals recomputed, exactly.** `attend_cached` does the same
arithmetic in the same order as the uncached attention. So decode output is
bit-identical to re-running the whole sequence, and the test asserts that at
every step. An equality test is far stronger than a tolerance test.

**11. Greedy decoding.** Always take the highest-scoring token. Simple and
deterministic, which makes it comparable with Hugging Face `generate` token
for token.

**12. Argmax on the rack.** The client wants one token, not 50,257 scores.
Each rank finds the best token in its own vocabulary slice and sends just
(score, id); the root keeps the best. 8 bytes per rank instead of 100 kB.

**13. Prefill and decode are different workloads.** Prefill: many tokens per
request, big matrices, throughput matters. Decode: one token per request,
tiny messages, latency matters. Here a decoded token cost about 4x a
prefill token.

**14. Resource limits and cleanup.** State that clients create must be
bounded and reclaimed. There is a session limit, and when a client's
connection closes the rack frees whatever it left open.

## Decisions

| Decision | Why | What was given up |
|----------|-----|-------------------|
| The client drives decoding one step at a time | The client can choose any sampling method; the rack stays simple | One round trip per token |
| Also offer argmax on the rack | For greedy decoding there is no reason to ship logits | Only greedy; other sampling still needs the logits |
| The client picks the session id | No extra reply field needed | The client must not reuse a live id (the rack rejects it) |
| Cache keys and values, not hidden states | They are exactly what later tokens read; nothing else is needed | Nothing |
| One cached-attention routine for both prefill and decode | One piece of arithmetic to get right, and bit-identical results | Prefill is not specially optimised |
| Same arithmetic order as the uncached path | Allows the exact-equality test | Rules out reordering for speed, until Part 6 decides otherwise |
| All validation on the root, before the broadcast | A bad decode never touches any rank's cache | Nothing |
| Root sends the expected position with each decode | Detects caches that have drifted apart | 4 bytes per request |
| Free a client's sessions when it disconnects | A crashed client cannot leak memory | The root must track who opened what |
| Fixed limit of 64 sessions | Bounded memory | A real server would size this from available memory |
| Cache grows as tokens arrive | No memory wasted on short conversations | Occasional reallocation as it grows |

## Measured (4-core laptop, Intel Core i5-10300H)

32-token prompt, then 32 decoded tokens:

| Model | Ranks | Prefill | Decode per token | of which head | In collectives | Data per step | Same token, no cache |
|-------|-------|---------|------------------|---------------|----------------|---------------|----------------------|
| Dense | 1 | 320 ms | 40.0 ms | 13.2 ms | 0 ms | 0 kB | 517 ms |
| Dense | 2 | 165 ms | 37.0 ms | 10.7 ms | 5.5 ms | 151 kB | 290 ms |
| MoE top-2 of 8 | 2 | 88 ms | 25.8 ms | 9.5 ms | 6.0 ms | 151 kB | 169 ms |

Cache size: 64 tokens x 768 floats x 4 bytes x 2 (key and value) x 12 layers
= 4.7 MB per session, summed over all ranks.

Three things the numbers show that are worth understanding:

- **Without the cache, cost grows with length.** Token 65 costs 517 ms
  because all 64 earlier tokens are re-run. With the cache it costs 40 ms,
  the same as token 2.
- **A decoded token is more expensive than a prefill token** (40 ms against
  10 ms), although it is the same arithmetic. The reason is memory, not
  arithmetic: one decoded token has to read every weight in the model once
  (495 MB), straight from main memory, whereas in prefill each weight matrix
  is used for many tokens in a row while it is still in the CPU cache. Part 6
  measures this directly.
- **A second rank nearly halves prefill (320 to 165 ms) but barely moves
  decode (40.0 to 37.0 ms).** Part 6 shows why: ranks on one machine share
  its memory, and decode is limited by memory.

## Questions worth being able to answer

- Why can a token's key and value be stored and never recomputed?
- What exactly does one decode step do in one layer?
- Without a cache, how does the cost of generating N tokens grow? With one?
- Why does each rank keep only part of the cache, and which part?
- Why is "bit-identical to recomputing" a better test than "close to
  PyTorch"?
- What new failure does keeping state introduce that a stateless request did
  not have?
- Why might one decoded token cost more than one prefill token?
