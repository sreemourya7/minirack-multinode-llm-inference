# minirack

A miniature multi-node LLM inference runtime, written from scratch in C++ with
a Python front end. Worker processes stand in for the nodes of a rack: each
holds one shard of the model, and they talk to each other over sockets.

The goal is to work through the problems a real inference runtime has to
solve (splitting a model across nodes, moving activations between them,
keeping per-request state, surviving a dead node, and finding where the time
goes) at a size that fits on a laptop.

## Results at a glance

Everything is checked against Hugging Face GPT-2 in PyTorch. Measured on a
2-core machine; details and caveats are in the Results section.

| | Result |
|---|---|
| Correctness | Logits within 3e-4 of PyTorch in every configuration; greedy text is token-for-token identical |
| Tensor parallelism | 2 ranks on 2 cores answer one request 1.8x to 2.0x faster |
| Mixture of Experts | Running 2 of 8 experts per token halves the work; every routing decision matches a PyTorch reference |
| KV cache | Bit-identical to recomputing from scratch; the 65th token costs about 23x less |
| Failure recovery | `kill -9` a rank mid-generation: the output is bit-identical and the client sees no error. With a standby and checkpoints that step takes 33 ms instead of 22 to 28 |
| AVX2 kernels | Prefill 5.5x to 6.2x faster (165 to 178 tokens/s on one rank, 318 to 378 on two). Decode 2.1x to 2.9x faster, within 6% to 16% of the memory roofline on one rank |
| Start to finish | Generating the 65th token: 2.4 s in the first version, 20 to 29 ms in the last |

![Time per token before and after the AVX2 kernels](docs/part6-profile.svg)

## Parts

| Part | What it adds | State |
|------|--------------|-------|
| 1 | GPT-2 forward pass in C++, split layer-wise across worker processes, verified against PyTorch | done |
| 2 | Collectives (broadcast, all-reduce, gather) and tensor parallelism: every rank works on every layer at once | done |
| 3 | A Mixture-of-Experts layer with experts sharded across ranks and per-token routing | done |
| 4 | KV cache, separate prefill and decode paths, per-request state | done |
| 5 | Failure detection and recovery when a rank dies or hangs mid-generation | done |
| 6 | AVX2 kernels for the hot paths, a profile of where the time goes, and the memory roofline | done |

## How it works, part by part

**Part 1, pipeline split (`mr_worker`).** Cut between layers. Workers take turns.

```
client --tokens--> worker 0 ----hidden [T x 768]----> worker 1
                   embed + layers 0-5                 layers 6-11 + head
client <--logits-- worker 0 <-------logits----------- worker 1
```

**Part 2, tensor parallelism (`mr_rank`).** Cut inside every layer. All ranks
work on the same layer at the same time, each on its own slice of the weights,
and combine their partial results with an all-reduce.

```
                  +--> rank 0: heads 0-5,  MLP units 0-1535    --+
 hidden state h --+                                              +--> all-reduce (sum) --> next layer
 (same on all)    +--> rank 1: heads 6-11, MLP units 1536-3071 --+
```

**Part 3, Mixture of Experts (`mr_rank --moe`).** The MLP becomes 8 small
experts. A router picks the best 2 for each token, and only those run. Each
rank owns some of the experts.

```
                         +--> rank 0 owns experts 0-3: runs the ones chosen, on the tokens that chose them --+
 token --> router: top 2 +                                                                                   +--> all-reduce
                         +--> rank 1 owns experts 4-7: same                                                  --+
```

The MoE here reuses GPT-2's weights (expert e is a group of GPT-2's MLP hidden
units) with a random, untrained router. It does not produce sensible text. It
exists to build and test the runtime side of MoE: routing, running only the
chosen experts, and the uneven load that results.

**Part 4, KV cache (`prefill` / `decode` / `release`).** Each token's keys and
values are computed once and kept, so a new token costs one token of work
instead of re-running the whole text. Each rank keeps the cache for its own
attention heads.

```
 prefill "Hello, my dog is"  --> run 4 tokens once, store their keys and values (a session)
 decode  " cute"             --> run 1 token: append its key and value, attend over the 5 stored
 decode  "."                 --> run 1 token: append, attend over the 6 stored
 release                     --> free the session on every rank
```

**Part 5, failure recovery.** A rank is killed or freezes while text is being
generated. The root notices, cancels the half-done step on the surviving
ranks, takes in a replacement, rebuilds the lost cache, and runs the step
again. The client sees a pause, no error, and exactly the same output.

```
 rank 1 dies       root notices: socket      replacement:         cache rebuilt:        the interrupted
 or hangs     -->  closed, or silent    -->  a waiting standby,   replay the tokens, --> step runs again
                   past the timeout          or a new process     or copy a checkpoint
```

![Rebuild time against session length for the two recovery strategies](docs/part5-recovery.svg)

**Part 6, kernels and profile.** Measure where the time goes, rewrite the
hot loops with AVX2, measure again. Prefill, where each weight matrix is
reused for many tokens while it sits in the CPU cache, became about 6x
faster. Decode, where one token has to read every weight once, became 2.1x to
2.9x faster and then ran into a wall that no arithmetic trick moves: the
speed at which the machine can read memory.

```
 time per decoded token  >=  bytes of weights read per token  /  memory read speed
                               495 MB                          /  13.6 GB/s        =  36 ms     (measured: 40 ms)
```

## Requirements

- Linux (on Windows, use WSL2). The networking code uses POSIX sockets.
- g++ with C++17, CMake 3.16+
- Python 3 with numpy
- For the correctness checks only: `pip install torch transformers`

## Quick start

```bash
# 1. Build
cmake -S . -B build && cmake --build build -j

# 2. Get GPT-2 small (124M) from Hugging Face and convert it
mkdir -p weights/gpt2 && cd weights/gpt2
curl -LO https://huggingface.co/openai-community/gpt2/resolve/main/config.json
curl -LO https://huggingface.co/openai-community/gpt2/resolve/main/model.safetensors
# optional, only for scripts/generate.py (text in, text out):
for f in vocab.json merges.txt tokenizer.json; do
  curl -LO https://huggingface.co/openai-community/gpt2/resolve/main/$f
done
cd ../..
python3 scripts/export_weights.py --safetensors weights/gpt2/model.safetensors \
    --config weights/gpt2/config.json --out weights/gpt2.mrkw

# 3. Check against PyTorch (each script starts and stops its own processes)
python3 tests/check_part1.py --weights weights/gpt2.mrkw --hf-dir weights/gpt2
python3 tests/check_part2.py --weights weights/gpt2.mrkw --hf-dir weights/gpt2
python3 tests/check_part3.py --weights weights/gpt2.mrkw --hf-dir weights/gpt2
python3 tests/check_part4.py --weights weights/gpt2.mrkw --hf-dir weights/gpt2
python3 tests/check_part5.py --weights weights/gpt2.mrkw --hf-dir weights/gpt2
python3 tests/check_part6.py --weights weights/gpt2.mrkw --hf-dir weights/gpt2
scripts/run_all_checks.sh weights/gpt2.mrkw weights/gpt2    # or: build and run all six

# Or run a rack by hand
scripts/run_local.sh weights/gpt2.mrkw        # Part 1: 2-worker pipeline
scripts/run_local_tp.sh weights/gpt2.mrkw 2   # Part 2: 2 tensor-parallel ranks

# Part 3: make an MoE file (8 experts, top-2), then run 2 ranks with it
python3 scripts/make_moe.py --base weights/gpt2.mrkw --experts 8 --top-k 2 --seed 8 --out weights/moe8.mrkw
scripts/run_local_tp.sh weights/gpt2.mrkw 2 weights/moe8.mrkw

# Part 4: with a (dense) rack running, generate text from another terminal
python3 scripts/generate.py --hf-dir weights/gpt2 --prompt "Hello, my dog is cute" --tokens 30

# Part 5: kill a rank mid-generation and compare with an undisturbed run (one command)
python3 scripts/demo_failover.py --weights weights/gpt2.mrkw --hf-dir weights/gpt2
python3 scripts/demo_failover.py --weights weights/gpt2.mrkw --hf-dir weights/gpt2 --recovery replay --no-standby
python3 scripts/demo_failover.py --weights weights/gpt2.mrkw --hf-dir weights/gpt2 --hang

# Or keep a rack running under the supervisor, generate from another terminal, and `kill -9` a rank yourself
python3 scripts/supervisor.py --weights weights/gpt2.mrkw --nranks 2 --standby --recovery checkpoint

# Part 6: kernel tests, the microbenchmark, and the same rack with the old scalar loops
build/mr_kernel_test
build/mr_bench
MR_KERNELS=scalar scripts/run_local_tp.sh weights/gpt2.mrkw 2
python3 scripts/plot_profile.py && python3 scripts/plot_recovery.py   # redraw the charts from your own run
```

## Results

All numbers are from the development machine: a 2-core virtual machine
(Intel Xeon, 2.1 GHz). Run the checks on your own machine for your own
numbers; `build/mr_bench` in particular depends on your CPU's caches and
memory. The reference is Hugging Face `GPT2LMHeadModel` on the same weights,
over six token sequences of length 1 to 32.

The tables for Parts 1 to 5 were measured with the scalar kernels, as each
part was built. `MR_KERNELS=scalar` reproduces them. The Part 6 section
compares scalar with AVX2 and shows how the earlier numbers moved. Timings
vary by roughly 10% to 15% between runs on this machine, so where a result
was measured more than once the range is given.

### Part 1: pipeline split

| Check | Result |
|-------|--------|
| Largest logit difference vs PyTorch, any position, any split | 3.05e-4 |
| Next-token prediction (argmax) matches PyTorch | every position |
| 2- and 3-worker output vs 1-worker output | bit-identical |
| Bad requests (out-of-range token, empty, too long) | error reply, workers keep serving |

A 32-token request with logits for every position takes 2.2 to 2.3 s, and
splitting it across workers does not change that beyond run-to-run noise. A
pipeline split does not make one request faster: the workers take turns. What
it buys is memory, and throughput once several requests are in flight.

### Part 2: tensor parallelism

| Check | Result |
|-------|--------|
| Largest logit difference vs PyTorch, 1 to 4 ranks | 3.2e-4 |
| Next-token prediction (argmax) matches PyTorch | every position |
| Same request twice | bit-identical |
| All ranks hold a bit-identical hidden state | checked on every request |
| One rank corrupts its hidden state (forced in the test) | caught by the replica check |
| One rank is killed and nothing replaces it | client gets an error reply after the configured wait, no hang |

Latency of one 32-token request (last-position logits, best of 5):

| Ranks | Latency | Speedup | Time in collectives (root) | Data through the root |
|-------|---------|---------|----------------------------|-----------------------|
| 1 | 1168 ms | 1.00x | 0 ms | 0 MB |
| 2 | 596 ms | 1.96x | 13 ms | 4.9 MB |
| 3 | 699 ms | 1.67x | 132 ms | 9.8 MB |
| 4 | 596 ms | 1.96x | 50 ms | 14.6 MB |

How to read this:

- Two ranks on two cores is 1.96x faster in this run (1.84x to 1.98x across
  seven runs). The 27 collectives of a request moved 4.9 MB and cost about 2% of
  the time.
- Three and four ranks are no faster, because the machine has two cores. The
  extra ranks share cores, and "time in collectives" grows because it includes
  waiting for a rank that is not currently running. This is a limit of the
  test machine, not a measurement of how the design scales.
- Memory: with 2 ranks, the root holds 328 MB of weights and rank 1 holds
  247 MB, against 498 MB for the whole model. The root is larger because it
  keeps the full token table for embedding.

### Part 3: Mixture of Experts

Two references are used. The anchor (8 experts, top-8, every gate equal to 1)
is mathematically GPT-2's own MLP, so it is compared with plain Hugging Face
GPT-2. The real MoE (8 experts, top-2, softmax gates) is compared with a
PyTorch MoE model: Hugging Face GPT-2 with each MLP replaced.

| Check | Result |
|-------|--------|
| Anchor vs plain GPT-2, 1 / 2 / 4 ranks: largest logit difference | 2.9e-4 |
| MoE vs PyTorch MoE reference, 1 to 4 ranks: largest logit difference | 2.8e-4 |
| Next-token prediction (argmax) matches | every position |
| Tokens per expert reported by the rack vs the reference | identical |
| (token, expert) pairs executed across all ranks | exactly layers x tokens x k |
| Same request twice | bit-identical |

Latency of one 32-token request (last-position logits, best of 5):

| Model | Ranks | Latency | Compute per rank | Expert work per rank |
|-------|-------|---------|------------------|----------------------|
| Dense | 1 | 1195 ms | 1195 ms | - |
| Dense | 2 | 606 ms | 586 / 598 ms | - |
| MoE top-2 of 8 | 1 | 635 ms | 635 ms | 100% |
| MoE top-2 of 8 | 2 | 356 ms | 337 / 287 ms | 61% / 39% |

How to read this:

- MoE is 1.88x to 2.00x faster than dense on one rank. Each token runs 2 of 8
  experts, so the MLP does a quarter of the work, and the MLP is about two
  thirds of a layer's arithmetic. Attention is unchanged.
- MoE on two ranks is 1.73x to 1.78x faster than on one, where dense reaches
  1.84x to 1.98x. The reason is load balance. The router sent 61% of the
  expert work to rank 0's experts, and the all-reduce makes everyone wait for
  the busiest rank.
- Tokens per expert for this request were 104, 173, 106, 89, 58, 59, 101, 78.
  Expert 1 got three times the load of expert 4.

Known limits at this stage: the router is random and untrained (see above),
experts are assigned to ranks as fixed contiguous ranges rather than by load,
and each rank still sends a full-size matrix to the all-reduce even when most
of its rows are zero.

### Part 4: KV cache

| Check | Result |
|-------|--------|
| Prefill + 12 decode steps vs re-running with no cache (dense on 1, 2, 4 ranks; MoE on 2) | bit-identical at every step |
| The same logits vs PyTorch | largest difference 2.9e-4 |
| Two sessions advanced in alternation vs each alone | bit-identical |
| Wrong uses (unknown session, duplicate id, bad token, too long, use after release) | error reply; the open session is left intact |
| Client disconnects without releasing | the rack frees its sessions on every rank |
| Session limit (64) | enforced |
| 24 greedy tokens vs Hugging Face `generate` | identical tokens |

Timing for a 32-token prompt followed by 32 decoded tokens (two runs; the
table shows the second):

| Model | Ranks | Prefill | Decode per token | of which head | In collectives | Data per step | Same token, no cache |
|-------|-------|---------|------------------|---------------|----------------|---------------|----------------------|
| Dense | 1 | 1323 ms | 105 ms | 33 ms | 0 ms | 0 kB | 2383 ms |
| Dense | 2 | 767 ms | 55 ms | 17 ms | 3.6 ms | 151 kB | 1261 ms |
| MoE top-2 of 8 | 2 | 365 ms | 41 ms | 17 ms | 4.3 ms | 151 kB | 776 ms |

How to read this:

- The cache makes token 65 about 23x to 25x cheaper on one rank (105 ms
  instead of about 2.4 s), and the gap widens with every token generated.
- A session of 64 tokens holds 4.7 MB of keys and values in total, split
  across the ranks by attention head.
- A decoded token costs about 2.6x what a prefill token costs (105 ms against
  about 40 ms). The likely reason, to be confirmed by the Part 6 profile: in
  prefill each weight matrix is reused for all 32 tokens while it sits in the
  CPU cache, but a single decoded token has to pull all 498 MB of weights
  from main memory once.
- Communication is 4% to 7% of a dense decode step and about 10% for MoE, up
  from about 2% in the Part 2 prefill-style request. The messages are small
  (27 collectives of about 3 kB each way) but the compute per step is small
  too.
- Picking the next token on the rack sends 8 bytes per rank instead of that
  rank's slice of the logits: 151 kB per step instead of 252 kB. On loopback
  the time was the same within noise (55 to 56 ms per token).
- Prefill on 2 ranks was 641 ms in one run and 767 ms in the other. It is a
  single measurement per run and should be read as roughly 1.7x to 2.0x
  faster than one rank.

### Part 5: failure recovery

Every scenario is run twice, once undisturbed and once with a failure, and the
logits must be bit-identical with no error reaching the client.

| Scenario | Result |
|----------|--------|
| Rank killed between two steps, two sessions open (replay and checkpoint) | both sessions continue, bit-identical |
| Rank killed in the middle of a prefill or a decode step (2 ranks) | step is run again, bit-identical |
| The same with 3 ranks, killing either non-root rank | the surviving rank drops the half-done step, falls back in step, bit-identical |
| Rank frozen with SIGSTOP (root timeout 1000 ms) | found by timeout, standby takes over, bit-identical; the frozen process exits by itself when woken |
| Rank killed and nothing replaces it | clear error after the configured wait; nothing half-applied; the session carries on once a replacement arrives |
| Three failures during one generation | bit-identical |
| Under the supervisor, killed at token 9 of 24 | text identical to Hugging Face `generate` |
| Mixture-of-Experts model | bit-identical |

What the client sees on the step that hits the failure (64-token session,
2 ranks; a normal step is 55 to 61 ms):

| Replacement | State rebuilt by | That step | of which waiting for a rank | of which rebuilding |
|-------------|------------------|-----------|-----------------------------|---------------------|
| new process (loads its weights) | replay | 1635 ms | 208 ms | 1368 ms |
| new process (loads its weights) | checkpoint | 274 ms | 211 ms | 1.8 ms |
| standby (weights already loaded) | replay | 1333 ms | 0 ms | 1279 ms |
| standby (weights already loaded) | checkpoint | 76 ms | 0 ms | 2.0 ms |

Time to rebuild one session on the replacement rank (the recovery chart near
the top; its third line, replay with the AVX2 kernels, is explained under Part 6):

| Tokens in the session | Replay | Checkpoint | Checkpoint copies |
|-----------------------|--------|------------|-------------------|
| 16 | 348 ms | 0.2 ms | 0.6 MB |
| 64 | 1341 ms | 2.0 ms | 2.4 MB |
| 128 | 2646 ms | 6.3 ms | 4.7 MB |
| 256 | 5763 ms | 11.7 ms | 9.4 MB |
| 512 | 11013 ms | 31.0 ms | 18.9 MB |

How to read this:

- With a standby and checkpoints, a `kill -9` costs the client about one
  extra step: 76 ms here, and 108 ms in the demo where the kill landed in the
  middle of a step that then had to be run again. With neither it costs
  1.6 s, and that grows with the session.
- Replay rebuilds at about 21 ms per token, so its cost follows the session
  length: 11 s at 512 tokens. Checkpoint copies bytes back: 31 ms at 512
  tokens, about 350 times faster.
- Checkpointing is not free. Every decode step ships 37 kB more (188 kB
  instead of 151 kB, 24% more), and the root holds a second copy of the other
  ranks' cache (37 kB per token here). Step time was 57 to 61 ms with either
  strategy across two runs, so on loopback the time cost is below the noise.
- A crash is noticed at once, because the dead process's socket closes. A
  frozen process keeps its socket open, so only the timeout reveals it: with
  a 1000 ms timeout that step took about 1070 ms.
- Exactness comes from Part 4. Replay works because prefill and step-by-step
  decode build bit-identical caches, so re-running the tokens recreates
  exactly what was lost.

Known limits of the recovery: the root itself is a single point of failure
(it holds the client connections, the token history and the checkpoints). A
frozen rank is only noticed while a request is running. The timeout must be
longer than the slowest legitimate step, and a rank that is merely slow past
it is replaced although it was healthy. Without a standby the supervisor
cannot replace a frozen process, because it never exits. If the replica check
fails (silent corruption) the rack still stops rather than repairs itself.

### Part 6: AVX2 kernels, the profile, and the memory roofline

| Check | Result |
|-------|--------|
| Kernel unit tests: AVX2 against scalar on 15 awkward shapes (sizes that are not multiples of 8 or 4, single rows, empty inputs) | agree to the last few bits |
| Batch invariance: T tokens in one call against one call per token, every kernel | bit-identical |
| End to end with each kernel set: against PyTorch, cached against recomputed, greedy text against Hugging Face | within 2.7e-4; bit-identical; identical |
| Mixture of Experts with each kernel set | every routing decision still matches the reference |
| A scalar rank tries to join an AVX2 rack | refused |
| All checks of Parts 1 to 5, rerun with the AVX2 kernels | pass |

Before and after (32-token prompt, then 32 decoded tokens; a 256-token prompt
for the throughput column):

| Model | Ranks | Kernels | Prefill, 32 tokens | Decode per token | Prefill throughput |
|-------|-------|---------|--------------------|------------------|--------------------|
| Dense | 1 | scalar | 1213 ms | 108.1 ms | 25 tokens/s |
| Dense | 1 | AVX2 | 202 ms | 40.0 ms | 177 tokens/s |
| | | speedup | 6.0x | 2.7x | 7.0x |
| Dense | 2 | scalar | 600 ms | 55.0 ms | 53 tokens/s |
| Dense | 2 | AVX2 | 96 ms | 22.8 ms | 377 tokens/s |
| | | speedup | 6.2x | 2.4x | 7.1x |
| MoE top-2 of 8 | 2 | scalar | 407 ms | 41.7 ms | 81 tokens/s |
| MoE top-2 of 8 | 2 | AVX2 | 74 ms | 18.5 ms | 595 tokens/s |
| | | speedup | 5.5x | 2.3x | 7.4x |

Two further full runs gave speedups of 5.5x to 6.2x for the 32-token prefill,
2.1x to 2.9x for decode, and 6.0x to 7.6x for prefill throughput.

Where one rank's time goes (dense, one rank, milliseconds per token; this is
the chart at the top):

| | Matrix multiplies | Vocabulary projection | GELU | Attention | Other | Total |
|---|---|---|---|---|---|---|
| Prefill, scalar | 35.74 | 0.98 | 0.85 | 0.25 | 0.08 | 37.90 |
| Prefill, AVX2 | 4.96 | 0.41 | 0.82 | 0.05 | 0.06 | 6.31 |
| Decode, scalar | 70.80 | 34.87 | 0.92 | 1.13 | 0.33 | 108.06 |
| Decode, AVX2 | 24.75 | 13.47 | 0.88 | 0.57 | 0.31 | 39.98 |

The matrix kernel on its own (`build/mr_bench`, one 768 x 3072 matrix,
billions of multiply-adds per second):

| Kernel | Weights in cache, 32 tokens per pass | Weights in cache, 1 token | Every weight read once | Memory read at |
|--------|--------------------------------------|---------------------------|------------------------|----------------|
| scalar | 2.65 | 2.68 | 1.32 | 5.3 GB/s |
| scalar, loops reordered | 2.35 | 2.61 | 1.25 | 5.0 GB/s |
| AVX2 | 21.76 | 7.44 | 3.67 | 14.7 GB/s |

For comparison, simply reading memory on this machine runs at 13.6 GB/s per
core.

The memory roofline for decode:

| Model | Ranks | Weights read per token, per rank | Time just to read them | Measured compute | Measured / roofline |
|-------|-------|----------------------------------|------------------------|------------------|---------------------|
| Dense | 1 | 495 MB | 36.3 ms | 40.0 ms | 1.10x |
| Dense | 2 | 247 MB | 18.2 ms | 20.8 ms | 1.15x |
| MoE top-2 of 8 | 2 | 162 MB (average) | 11.9 ms | 15.5 ms | 1.30x |

In two further runs these ratios were 1.06x to 1.16x, 1.01x to 1.41x, and
1.17x to 1.32x.

Compute against communication (dense, 2 ranks, AVX2): copying messages took
3.7 ms of a 96 ms prefill (4%) and 1.6 ms of a 22.8 ms decode step (7%). With
the scalar kernels the shares were 1% and 4%.

How to read this:

- **Prefill is limited by arithmetic, so SIMD pays in full.** With the
  weights in cache the AVX2 kernel does 21.8 billion multiply-adds per second
  against 2.7 for the scalar loop, 8 times more. End to end that is 5.5x to
  6.2x, because the parts that were not rewritten now matter.
- **Decode is limited by memory.** One decoded token must read every weight
  once, 495 MB, and nothing is reused. The AVX2 kernel in that situation
  reads weights at 14.7 GB/s, which is as fast as this machine reads memory
  at all. So decode sits within 6% to 16% of the time needed just to read
  the weights. A faster kernel would not help. Reading fewer bytes per token
  would.
- **That explains the question left open in Part 4**, why a decoded token
  costs several times a prefill token for the same arithmetic. It was already
  true for the scalar code (2.7 against 1.3 billion multiply-adds per second,
  cache against memory), and it is much more visible now: 6.3 ms against
  40 ms.
- **Both ways of reading fewer bytes per rank show up in the numbers.** Two
  ranks each read half the weights, and decode goes from 40.0 to 22.8 ms. The
  Mixture-of-Experts model reads only the chosen experts, and decodes in
  18.5 ms.
- **Reordering the loops alone did nothing for the scalar kernel** (2.35
  against 2.65). Reusing a row of weights while it is in cache only pays once
  the arithmetic is fast enough for memory to be the bottleneck.
- **What was not rewritten is now visible.** GELU was 2% of a prefill token
  and is 13% after the speedup. It was left as it is because a vector version
  needs an approximation of `tanh`, and this project keeps exact agreement
  with the standard library.
- **Communication is still small here, 4% to 7%**, but its share roughly
  doubled when compute got faster. A decode step makes 27 round trips to the
  root whatever the model size. On hardware where the arithmetic takes
  microseconds, those round trips would be most of the step.

Faster prefill also changed Part 5, because recovery by replay *is* a prefill:

| | Scalar kernels | AVX2 kernels |
|---|---|---|
| Replay rebuild, 512-token session | 11.0 s | 1.7 s |
| Checkpoint rebuild, 512-token session | 31 ms | 28 ms |
| Step that hit a `kill -9`, new process + replay | 1635 ms | 562 ms |
| Step that hit a `kill -9`, standby + checkpoint | 76 ms | 33 ms |
| A normal decode step, for comparison | 55 to 61 ms | 22 to 28 ms |

Checkpointing still ships 24% more bytes per step. Its effect on step time
was again below the run-to-run noise (four runs each: 21.9 to 28.0 ms with
replay, 21.9 to 24.1 ms with checkpoint).

Design rules the kernels keep: the result for one token never depends on the
other tokens in the batch (that is what keeps decode equal to prefill, and
replay exact), only one file is compiled with AVX2 flags and it is entered
only after a run-time check of the CPU, and a rack refuses to mix kernel
sets, because a replacement rank must reproduce the arithmetic of the rank it
replaces.

## Limits, and where the measurements point next

- The root is a single point of failure, and it serves one client at a time.
- Decode is at the memory roofline, so the next gain there is fewer bytes per
  token: 16-bit or 8-bit weights would halve or quarter what each token
  reads. Batching several sessions through one pass over the weights would
  share that cost between them.
- GELU is the next hotspot in prefill (13%).
- Every message goes through the root. With more ranks a ring or tree
  all-reduce would spread that load.
- Experts are placed on ranks in fixed ranges. Placing them by measured load
  would even out the 61% / 39% split seen in Part 3.
- The Mixture-of-Experts router is random and untrained: that model exists to
  exercise the runtime, not to produce sensible text.
- The AVX2 kernels need an x86-64 CPU with AVX2 and FMA. Elsewhere the scalar
  kernels are used automatically.

## Layout

```
src/mat.h          float32 matrix
src/ops.*          linear, matmul, layernorm, GELU, causal attention
src/weights.*      .mrkw weight file reader (whole tensors or row slices)
src/net.*          sockets and the framed wire protocol
src/model.*        Part 1 shard: embedding? + layers A..B + head?
src/worker.cpp     Part 1 process: one stage of the pipeline
src/mesh.*         Part 2 collectives: broadcast, all-reduce, gather, replica check; Part 5 abort and rejoin
src/tp_model.*     Part 2 shard: a slice of every layer
src/moe.*          Part 3 router, top-k, gates, and sparse expert execution
src/kernels.*      Part 6 the hot loops: scalar versions, and run-time choice of implementation
src/kernels_avx2.cpp   Part 6 the AVX2 + FMA versions (the only file built with -mavx2 -mfma)
src/prof.*         Part 6 per-operation timers
src/bench.cpp      Part 6 microbenchmark: kernel speed hot and cold, memory read speed
src/kernel_test.cpp    Part 6 kernel unit tests
src/kv_cache.*     Part 4 sessions, the KV cache, and attention that uses it
src/exec.*         the program every rank runs for one request
src/root.cpp       rank 0: front door, event loop, and Part 5 failure recovery
src/rank.cpp       main(), options, and the loop of every non-root rank
scripts/           weight export, MoE file generator, Python client, text generation demo,
                   supervisor, failover demo, chart scripts, local launchers, run_all_checks.sh
tests/             correctness checks against PyTorch
docs/              per-part notes (concepts used and decisions taken), the two charts,
                   and docs/data/ with the measurements behind them
```
