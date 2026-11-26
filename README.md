# minirack

A miniature multi-node LLM inference runtime, written from scratch in C++ with
a Python front end. Worker processes stand in for the nodes of a rack: each
holds one shard of the model, and they talk to each other over sockets.

The goal is to work through the problems a real inference runtime has to
solve (splitting a model across nodes, moving activations between them,
keeping per-request state, surviving a dead node, and finding where the time
goes) at a size that fits on a laptop.

## Status

| Part | What it adds | State |
|------|--------------|-------|
| 1 | GPT-2 forward pass in C++, split layer-wise across worker processes, verified against PyTorch | done |
| 2 | Collectives (broadcast, all-reduce, gather) and tensor parallelism: every rank works on every layer at once | done |
| 3 | A Mixture-of-Experts layer with experts sharded across ranks and per-token routing | done |
| 4 | KV cache, separate prefill and decode paths, per-request state | planned |
| 5 | Failure detection and recovery when a rank dies mid-generation | planned |
| 6 | AVX2 kernels for the hot paths, and a profile of compute versus communication | planned |

## Two ways to split a model

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
cd ../..
python3 scripts/export_weights.py --safetensors weights/gpt2/model.safetensors \
    --config weights/gpt2/config.json --out weights/gpt2.mrkw

# 3. Check against PyTorch (each script starts and stops its own processes)
python3 tests/check_part1.py --weights weights/gpt2.mrkw --hf-dir weights/gpt2
python3 tests/check_part2.py --weights weights/gpt2.mrkw --hf-dir weights/gpt2
python3 tests/check_part3.py --weights weights/gpt2.mrkw --hf-dir weights/gpt2

# Or run a rack by hand
scripts/run_local.sh weights/gpt2.mrkw        # Part 1: 2-worker pipeline
scripts/run_local_tp.sh weights/gpt2.mrkw 2   # Part 2: 2 tensor-parallel ranks

# Part 3: make an MoE file (8 experts, top-2), then run 2 ranks with it
python3 scripts/make_moe.py --base weights/gpt2.mrkw --experts 8 --top-k 2 --seed 8 --out weights/moe8.mrkw
scripts/run_local_tp.sh weights/gpt2.mrkw 2 weights/moe8.mrkw
```

## Results

All numbers below are from the development machine: 2 CPU cores, plain scalar
`-O2` code, no KV cache. Run the checks on your own machine for your own
numbers. The reference is Hugging Face `GPT2LMHeadModel` on the same weights,
over six token sequences of length 1 to 32.

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
| One rank is killed | client gets an error reply, no hang |

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

Known limits overall: the code is a deliberately plain baseline (speed
is Part 6), and a lost rank takes the rack down until it is restarted
(recovery is Part 5).

## Layout

```
src/mat.h          float32 matrix
src/ops.*          linear, matmul, layernorm, GELU, causal attention
src/weights.*      .mrkw weight file reader (whole tensors or row slices)
src/net.*          sockets and the framed wire protocol
src/model.*        Part 1 shard: embedding? + layers A..B + head?
src/worker.cpp     Part 1 process: one stage of the pipeline
src/mesh.*         Part 2 collectives: broadcast, all-reduce, gather, replica check
src/tp_model.*     Part 2 shard: a slice of every layer
src/moe.*          Part 3 router, top-k, gates, and sparse expert execution
src/rank.cpp       Part 2-3 process: one tensor-parallel rank
scripts/           weight export, MoE file generator, Python client, local launchers
tests/             correctness checks against PyTorch
docs/              per-part notes: concepts used and decisions taken
```
