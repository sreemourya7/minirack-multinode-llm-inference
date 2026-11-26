# Part 6 notes: concepts and decisions

Parts 1 to 5 made the runtime correct, parallel, stateful and resilient, on
purpose with plain loops. Part 6 asks the last question: **where does the
time actually go, and what is the limit?** The method is the result as much
as the numbers: measure, change one thing, measure again, and explain what
the measurements say.

```
 time per decoded token  >=  bytes of weights read per token  /  memory read speed
                               495 MB                          /  about 15 GB/s    =  about 33 ms
 measured: 43.9 ms on one rank; 32.5 ms of compute (38.5 ms in all) on two
```

## Concepts

**1. Profile before optimising.** Every basic operation adds its time to a
counter (`src/prof.h`). The first profile said 95% of a prefill was in the
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

**10. Why prefill and decode differ.** In prefill, a few rows of a weight
matrix are used for a whole block of tokens before moving on: hot. In decode,
one token passes through every matrix once, 495 MB in total, and by the time
a matrix is needed again it is long gone from the cache: cold. Same
arithmetic, very different memory behaviour. Measured: 10.8 billion
multiply-adds per second hot, 3.4 cold.

**11. Compute-bound versus memory-bound.** A loop is compute-bound when the
arithmetic is the limit, and memory-bound when fetching the data is. Prefill
is compute-bound, so faster arithmetic helped in full (about 9x). Decode is
memory-bound, so it helped less (about 3x) and then stopped.

**12. Roofline.** A bound that comes from the hardware, not from the code.
Here: a decoded token must read every weight once, so it cannot take less
than (bytes of weights) / (memory read speed).

**13. Memory is shared by all the cores of one machine.** This is the
finding that matters most on a laptop. One process reads weights at
15.7 GB/s. Two at once get 7.1 GB/s each, four get 3.7 GB/s each. The total
stays near 15 GB/s. So a second rank on the same machine brings a second
core, but no second memory.

**14. What that does to scaling.** Prefill is compute-bound, so a second rank
nearly halves it (257 to 136 ms). Decode is memory-bound, so a second rank
barely moves it (43.9 to 38.5 ms): the two ranks together read at about
15 GB/s, the same as the whole machine can deliver. In a real rack every
node has its own memory, so this limit applies to each node separately.

**15. What moves a memory roofline.** Not better arithmetic, and not more
cores on the same memory. Fewer bytes per token (smaller weights, or a model
that touches fewer of them, like MoE), or more memory (more machines).

**16. Amdahl's law.** Speeding up one part makes the untouched parts matter.
GELU was 1.6% of a prefill token and is now 14%, without changing at all.

**17. The same holds for communication.** Copying messages was 1% of a
prefill and 5% of a decode step; it is now 5% and 15%. The round trips did
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
| Loop over weight rows on the outside, tokens inside, in blocks of 16 | Each row is fetched once and reused for the whole block | Nothing for one token |
| Leave GELU scalar | A vector version needs an approximate `tanh`; this project keeps exact agreement with the standard library | 14% of prefill time is still there to take |
| Leave attention's softmax scalar | It is under 2% of the time | Nothing worth having |
| Measure memory speed with the kernel's own access pattern, and with 1, 2 and 4 processes at once | A simple one-pass read test turned out to under-report what the kernel achieves, and it says nothing about sharing | A longer benchmark |
| Report what was measured, including where it does not fit | One rank alone decodes 1.4x slower than its floor, and that gap is not explained yet | Less tidy conclusions |

## Measured (4-core laptop, Intel Core i5-10300H)

Before and after:

| Model | Ranks | Kernels | Prefill, 32 tokens | Decode per token | Prefill throughput |
|-------|-------|---------|--------------------|------------------|--------------------|
| Dense | 1 | scalar | 2246 ms | 125.7 ms | 11 tokens/s |
| Dense | 1 | AVX2 | 257 ms | 43.9 ms | 119 tokens/s |
| Dense | 2 | scalar | 1090 ms | 66.3 ms | 27 tokens/s |
| Dense | 2 | AVX2 | 136 ms | 38.5 ms | 250 tokens/s |
| MoE top-2 of 8 | 2 | scalar | 616 ms | 51.2 ms | 48 tokens/s |
| MoE top-2 of 8 | 2 | AVX2 | 87 ms | 27.2 ms | 438 tokens/s |

The matrix kernel alone, billions of multiply-adds per second:

| Kernel | In cache, 32 tokens | In cache, 1 token | Every weight read once |
|--------|---------------------|-------------------|------------------------|
| scalar | 1.29 | 1.29 | 1.28 |
| scalar, loops reordered | 1.52 | 1.33 | 1.22 |
| AVX2 | 10.81 | 5.44 | 3.41 |

Reading weights in several processes at once:

| Processes | Each reads | All together |
|-----------|------------|--------------|
| 1 | 15.7 GB/s | 15.7 GB/s |
| 2 | 7.1 GB/s | 14.1 GB/s |
| 4 | 3.7 GB/s | 14.9 GB/s |

The roofline for decode:

| Model | Ranks | Read per token, per rank | Read speed per rank | Time just to read | Measured compute |
|-------|-------|--------------------------|---------------------|-------------------|------------------|
| Dense | 1 | 495 MB | 15.8 GB/s | 31.4 ms | 43.9 ms |
| Dense | 2 | 247 MB | 7.1 GB/s | 35.0 ms | 32.5 ms |
| MoE top-2 of 8 | 2 | 162 MB | 7.1 GB/s | 22.9 ms | 21.7 ms |

With two ranks the measured time is slightly under the "time just to read".
The two come from separate runs, so read it as "at the limit", not "beyond
it". With one rank, decode is 1.4x above its floor: a single rank reads at
about 11 GB/s during a real decode step against 15.7 GB/s in the benchmark,
and the reason for that gap has not been tracked down.

## Summary

The profile came first: 95% of prefill was matrix multiplies. That kernel was
rewritten with AVX2 and FMA, with run-time dispatch and tests that it stays
bit-identical across batch sizes. Prefill became about 9x faster. Decode
became only about 3x faster, and a microbenchmark showed why: one decoded
token has to read all 495 MB of weights once, and this laptop moves about
15 GB/s from memory in total, however many cores are reading. That puts a
floor near 33 ms per token. It also explains why a second rank nearly halves
prefill but barely helps decode on one machine: the ranks share the same
memory. So the next gain for decode is fewer bytes per token, or more memory
bandwidth, which is what quantisation, Mixture of Experts and more machines
each provide.

The points underneath that result:
- Why prefill and decode behave so differently for identical arithmetic.
- Why adding a rank helps prefill but not decode on one machine, and why it
  would help on a real rack.
- Why FMA changes the last bits, and which properties had to survive that
  (batch invariance, and no mixing of kernel sets in one rack).
- What the 27 round trips per decode step (about 6 ms here, 15% of the step)
  would mean on hardware where arithmetic takes microseconds.
- What is not explained: the 1.4x gap between one rank's decode and its
  floor.

## Questions worth being able to answer

- What did the first profile say, and how did it decide what to rewrite?
- Why does the AVX2 kernel process four rows of the weight matrix at a time?
- Why are the AVX2 results not bit-identical to the scalar ones? What must
  still be bit-identical, and why?
- Prefill got 9x faster and decode 3x, with the same kernel. Why?
- What is a roofline, and how is it computed for this model on this laptop?
- One process reads 15.7 GB/s, two read 7.1 GB/s each. What does that tell
  you about where a second rank can and cannot help?
- Decode is limited by memory. Name three ways to make it faster anyway.
- Why did GELU's share of the time go up when nobody touched it?
- The first version of the memory test reported 10.8 GB/s while the kernel
  was reading at 15.8 GB/s. What does that say about trusting a benchmark?
