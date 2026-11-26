# Part 3 notes: concepts and decisions

Part 2 made every rank do a share of the work for every token. Part 3 asks:
**what if most of the work can be skipped?** A Mixture-of-Experts (MoE) layer
runs only a few small "expert" MLPs per token. That saves compute, and it
brings two new runtime problems: routing and uneven load.

```
                         +--> rank 0 owns experts 0-3: runs the ones chosen, on the tokens that chose them --+
 token --> router: top 2 +                                                                                   +--> all-reduce
                         +--> rank 1 owns experts 4-7: same                                                  --+
```

## Concepts

**1. Expert.** A small MLP. Here the dense MLP's 3072 hidden units are cut
into 8 experts of 384 units each. Code: `Expert` in `src/moe.h`.

**2. Router.** One small matrix, `[768, 8]`. For each token it gives one score
per expert: `scores = x @ router_w + router_b`.

**3. Top-k.** Keep the k highest-scoring experts for the token (k = 2 here)
and ignore the rest. Different tokens choose different experts.

**4. Gate.** How much each chosen expert counts: a softmax over just the k
chosen scores, so the gates add up to 1.
Layer output = sum over chosen experts of `gate * expert(x)`, plus one shared
bias. Code for 2 to 4: `moe_route` in `src/moe.cpp`.

**5. Sparse execution.** An expert runs only on the tokens that chose it.
Three steps: *gather* those tokens' rows into a small matrix, *run* the
expert on it, *scatter* the results back to the right rows.
Code: `moe_run_local`. This is where the saving comes from: with top-2 of 8,
the MLP does a quarter of the arithmetic.

**6. Expert sharding.** Each rank owns some of the experts and loads only
their weights. A rank does work for a token only if the token chose one of
its experts.

**7. Replicated routing.** Every rank computes the routing for all tokens
itself. The router is tiny and every rank already has the same `x`, so they
all reach the same answer with no messages.

**8. Routing is a cliff.** Most of the model is smooth: a tiny change in the
input makes a tiny change in the output. Top-k is not. If the 2nd and 3rd
best scores are nearly tied, a change in the last bit can pick a different
expert and change the output a lot. Two consequences:
- The ranks' copies of `h` must be bit-identical (Part 2, concept 8).
  Otherwise two ranks could route the same token differently, and an expert
  would run twice or not at all.
- When comparing against PyTorch, near-ties make the comparison fragile.
  Over 16 router seeds, the closest call on the test set ranged from 3e-6 to
  5e-4. The test uses a seed whose closest call is 4e-4.

**9. Load imbalance.** The router does not spread tokens evenly. In the timed
request the experts received 104, 173, 106, 89, 58, 59, 101 and 78 tokens.
Rank 0's four experts got 61% of the work.

**10. The slowest rank sets the pace.** An all-reduce cannot finish until
every rank has sent its part. So the rank with the most work decides the
latency, and the others wait. This is why MoE on 2 ranks was 1.40x faster
than on 1, while dense reached 1.86x.

**11. Anchor test.** A special case with a known answer. With all 8 experts
chosen and every gate equal to 1, the MoE layer is exactly GPT-2's MLP. So
that configuration is checked against plain Hugging Face GPT-2, which no code
of ours touches.

**12. Conservation check.** Across all ranks, the number of (token, expert)
pairs executed must equal layers x tokens x k. The test checks it on every
request. Too many means an expert ran twice; too few means one was skipped.

## Decisions

| Decision | Why | What was given up |
|----------|-----|-------------------|
| Experts are slices of GPT-2's own MLP | Gives the anchor test: a real, external reference for the expert path | The MoE is not a trained model and does not produce sensible text |
| Random, untrained router | The subject is the runtime, not model quality | No claims about output quality can be made |
| Every rank computes the routing | No messages needed; rests on an invariant Part 2 already enforces and checks | Relies on all ranks running the same code on the same kind of CPU |
| MoE settings live in the sidecar file, not on the command line | Every rank reads the same file, so they cannot disagree on k or the gate type | One more file to pass around |
| Small sidecar file for the router | 296 kB instead of a second 498 MB copy of the model | Two files must be kept matching (the loader checks the model shape) |
| Contiguous expert ranges per rank | Simple; a rank's experts are one contiguous slice on disk | Load is whatever the router gives; no balancing |
| Reuse the Part 2 all-reduce unchanged | The MoE output is still a sum of parts | A rank sends a full `[T, 768]` matrix even if most rows are zero |
| Ties go to the lower expert id | Makes routing a pure function of the input bits | Nothing |
| PyTorch reference computes densely and masks | The opposite method to the C++ (which skips unchosen experts); two routes to the same numbers | The reference is our own code, which is why the anchor test also exists |
| Fix the router seed by its routing margin | Keeps the comparison stable across machines | The test covers one router, not many |

## Measured (4-core laptop, Intel Core i5-10300H; 32-token request)

| Model | Ranks | Latency | Compute per rank | Expert work per rank |
|-------|-------|---------|------------------|----------------------|
| Dense | 1 | 256 ms | 256 ms | - |
| Dense | 2 | 138 ms | 133 / 129 ms | - |
| MoE top-2 of 8 | 1 | 166 ms | 166 ms | 100% |
| MoE top-2 of 8 | 2 | 118 ms | 103 / 90 ms | 61% / 39% |

Why MoE is 1.5x and not 4x faster than dense: only the MLP shrinks. Per
token, a dense layer does about 2.4M multiply-adds in attention's matrices
and 4.7M in the MLP. Cutting the MLP to a quarter leaves 2.4M + 1.2M = 3.6M,
which is about half of 7.1M. So the arithmetic alone would give about 2x, and
with the slower scalar kernels that is roughly what is measured (1.8x, in the
Part 6 table). With the AVX2 kernels the matrix multiplies are fast enough
that what MoE does not shrink (the vocabulary projection, GELU, routing, and
gathering and scattering tokens) is a larger share, and the gain is 1.54x.

## Questions worth being able to answer

- What are the four steps of an MoE layer for one token?
- Why is MoE about 1.5x faster than dense here, when it does a quarter of the
  MLP's work?
- Why do the ranks not need to send each other the routing decisions? What
  would break that?
- Why did 2 ranks give less speedup for MoE than for dense?
- How would you reduce the load imbalance without retraining the router?
  (Think about which experts live on which rank.)
- What does the anchor test prove that the PyTorch MoE reference cannot?
