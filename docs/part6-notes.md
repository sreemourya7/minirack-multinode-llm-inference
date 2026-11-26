# Part 6 notes: concepts and decisions

Parts 1 to 5 made the runtime correct, parallel, stateful and resilient, on
purpose with plain loops. Part 6 asks the last question: **where does the
time actually go, and what is the limit?** The method is the result as much
as the numbers: measure, change one thing, measure again, and explain what
the measurements say.

```
 time per decoded token  >=  bytes of weights read per token  /  memory read speed
                               495 MB                          /  13.6 GB/s        =  36 ms     (measured: 40 ms)
```

## Concepts

**1. Profile before optimising.** Every basic operation adds its time to a
counter (`src/prof.h`). The first profile said 94% of a prefill was in the
matrix multiplies. So that is the loop worth rewriting, and nothing else is.

**2. SIMD.** One instruction working on several numbers at once. AVX2
registers hold 8 floats. Code: `src/kernels_avx2.cpp`.

**3. FMA (fused multiply-add).** `a * b + c` as one instruction with one
rounding instead of two. Faster, and slightly different in the last bits. So
the AVX2 kernels do not match the scalar ones exactly, only to about 1e-6.

**4. The kernel trick used here.** The matrix loop takes four rows of the
weight matrix at a time. Each group of 8 outputs is loaded once, updated by
four multiply-adds in a row, and stored once, instead of being loaded and
stored four times.

**5. Independent accumulators.** A running sum makes every addition wait for
the one before it. The dot product keeps four running sums so the CPU can
overlap them, and adds them together at the end.

**6. Leftovers.** When a size is not a multiple of 8, the last few elements
are done one at a time. That code is where vector kernels usually go wrong,
so the unit tests use deliberately awkward sizes.

**7. Run-time dispatch.** The program checks the CPU when it starts and picks
the scalar or AVX2 kernels. Only one source file is compiled with AVX2 flags
and it is only entered after that check, so the same binary still runs on a
CPU without AVX2. Code: `kern::select_from_env` in `src/kernels.cpp`.

**8. Batch invariance.** The result for one token must not depend on which
other tokens are in the batch. Every kernel keeps this rule and the tests
check it bit for bit. It is what keeps decode equal to prefill (Part 4) and
recovery by replay exact (Part 5), with any kernel.

**9. Cache-hot versus cache-cold.** The CPU keeps recently used data in small
fast memories (caches). Data that is reused soon is "hot" and nearly free to
read. Data last touched long ago has to come from main memory, which is far
slower.

**10. Why prefill and decode differ.** In prefill, one weight matrix is used
for many tokens in a row: hot. In decode, one token passes through every
matrix once, 495 MB in total, and by the time a matrix is needed again it is
long gone from the cache: cold. Same arithmetic, very different memory
behaviour. Measured: 21.8 billion multiply-adds per second hot, 3.7 cold.

**11. Compute-bound versus memory-bound.** A loop is compute-bound when the
arithmetic is the limit, and memory-bound when fetching the data is. Prefill
is compute-bound, so faster arithmetic helped in full (about 6x). Decode is
memory-bound, so it helped less (about 2.5x) and then stopped.

**12. Roofline.** A bound that comes from the hardware, not from the code.
Here: a decoded token must read every weight once, so it cannot take less
than (bytes of weights) / (memory read speed). After Part 6 decode is within
6% to 16% of that bound on one rank (three runs).

**13. What moves a memory roofline.** Not better arithmetic. Fewer bytes per
token (smaller weights, or a model that touches fewer of them, like MoE), or
more memory channels (more ranks, each reading its own part).

**14. Amdahl's law.** Speeding up one part makes the untouched parts matter.
GELU was 2% of a prefill token and is now 13%, without changing at all.

**15. The same holds for communication.** Copying messages was 1% of a
prefill and 4% of a decode step; it is now 4% and 7%. The round trips did
not get slower. Everything around them got faster.

## Decisions

| Decision | Why | What was given up |
|----------|-----|-------------------|
| Hand-written intrinsics, not compiler flags like `-O3 -march=native` | Control over exactly what runs, and one binary for every x86-64 CPU | More code, and it must be tested carefully |
| Keep the scalar kernels as they were | An honest baseline, and a fallback for other CPUs | Two implementations to keep in agreement |
| Choose kernels at start-up, not at build time | The same binary runs everywhere; `MR_KERNELS=scalar` gives an instant A/B comparison | One indirect call per kernel call (negligible here) |
| Accept that AVX2 results differ from scalar in the last bits | FMA is where much of the speed comes from | Results are tied to the kernel set |
| Require batch invariance from every kernel | Decode = prefill and exact replay depend on it | Rules out tricks that would treat tokens differently depending on batch size |
| A rack refuses a rank with different kernels | A replacement must reproduce the lost rank's arithmetic exactly | A rack cannot mix CPU generations unless all run scalar |
| Loop over weight rows on the outside, tokens inside, in blocks of 16 | Each row is fetched once and reused for the whole block | Nothing for one token; for the scalar kernel it brought no gain at all |
| Leave GELU scalar | A vector version needs an approximate `tanh`; this project keeps exact agreement with the standard library | 13% of prefill time is still there to take |
| Leave attention's softmax scalar | It is under 2% of the time | Nothing worth having |
| Measure the memory read speed on the machine itself | The roofline is a property of this machine, not a constant | The numbers do not transfer to other machines |
| Report ranges where runs differ | Timings vary by 10% to 15% between runs here | Less tidy tables |

## Measured (2-core development machine)

Before and after:

| Model | Ranks | Kernels | Prefill, 32 tokens | Decode per token | Prefill throughput |
|-------|-------|---------|--------------------|------------------|--------------------|
| Dense | 1 | scalar | 1213 ms | 108.1 ms | 25 tokens/s |
| Dense | 1 | AVX2 | 202 ms | 40.0 ms | 177 tokens/s |
| Dense | 2 | scalar | 600 ms | 55.0 ms | 53 tokens/s |
| Dense | 2 | AVX2 | 96 ms | 22.8 ms | 377 tokens/s |
| MoE top-2 of 8 | 2 | scalar | 407 ms | 41.7 ms | 81 tokens/s |
| MoE top-2 of 8 | 2 | AVX2 | 74 ms | 18.5 ms | 595 tokens/s |

The matrix kernel alone, billions of multiply-adds per second:

| Kernel | In cache, 32 tokens | In cache, 1 token | Every weight read once |
|--------|---------------------|-------------------|------------------------|
| scalar | 2.65 | 2.68 | 1.32 |
| scalar, loops reordered | 2.35 | 2.61 | 1.25 |
| AVX2 | 21.76 | 7.44 | 3.67 (14.7 GB/s; the machine reads memory at 13.6 GB/s) |

The roofline for decode:

| Model | Ranks | Read per token, per rank | Roofline | Measured |
|-------|-------|--------------------------|----------|----------|
| Dense | 1 | 495 MB | 36.3 ms | 40.0 ms |
| Dense | 2 | 247 MB | 18.2 ms | 20.8 ms |
| MoE top-2 of 8 | 2 | 162 MB | 11.9 ms | 15.5 ms |

## Questions worth being able to answer

- What did the first profile say, and how did it decide what to rewrite?
- Why does the AVX2 kernel process four rows of the weight matrix at a time?
- Why are the AVX2 results not bit-identical to the scalar ones? What must
  still be bit-identical, and why?
- Prefill got 6x faster and decode 2.5x, with the same kernel. Why?
- What is a roofline, and how would you compute it for this model on your
  own laptop?
- Decode is at the memory roofline. Name three ways to make it faster anyway.
- Why did GELU's share of the time go up when nobody touched it?
- Your laptop has a much smaller cache than the machine these numbers came
  from. Which of the results do you expect to change, and in which direction?
