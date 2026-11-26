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
| 2 | Collectives (all-gather, all-reduce) and a Mixture-of-Experts layer with experts sharded across workers | planned |
| 3 | KV cache, separate prefill and decode paths, per-request state | planned |
| 4 | Failure detection and recovery when a worker dies mid-generation | planned |
| 5 | AVX2 kernels for the hot paths, and a profile of compute versus communication | planned |

## Part 1 in one picture

```
client --tokens--> worker 0 ----hidden [T x 768]----> worker 1
                   embed + layers 0-5                 layers 6-11 + head
client <--logits-- worker 0 <-------logits----------- worker 1
```

Any contiguous split works (`--layers A:B`), with any number of workers.

## Requirements

- Linux (on Windows, use WSL2). The networking code uses POSIX sockets.
- g++ with C++17, CMake 3.16+
- Python 3 with numpy
- For the correctness check only: `pip install torch transformers`

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

# 3. Check against PyTorch (starts and stops the workers itself)
python3 tests/check_part1.py --weights weights/gpt2.mrkw --hf-dir weights/gpt2

# Or run a 2-worker rack by hand
scripts/run_local.sh weights/gpt2.mrkw
```

## Part 1 results

Reference: Hugging Face `GPT2LMHeadModel` on the same weights. Six token
sequences (lengths 1 to 32), four ways of splitting the 12 layers.

| Check | Result |
|-------|--------|
| Largest logit difference vs PyTorch, any position, any split | 3.05e-4 |
| Next-token prediction (argmax) matches PyTorch | every position |
| 2- and 3-worker output vs 1-worker output | bit-identical |
| Bad requests (out-of-range token, empty, too long) | error reply, workers keep serving |

Timing on the machine this was developed on (2 vCPUs, scalar `-O2` code, no
KV cache): about 70 ms per token position, for example 2.2 to 2.3 s for a
32-token sequence. Splitting across 2 or 3 workers on one machine moved that
by a few percent in either direction, which is within run-to-run noise. Run the check on your own machine
for your own numbers.

Two things these numbers do not claim:

- A layer-wise split does not make a single request faster. The workers take
  turns. What it buys is memory (a model too big for one node) and, once
  several requests are in flight, throughput.
- The code is a deliberately plain baseline. Speed is the subject of Part 5.

## Layout

```
src/mat.h          float32 matrix
src/ops.*          linear, layernorm, GELU, causal attention
src/weights.*      .mrkw weight file reader (loads one shard)
src/model.*        shard = embedding? + layers A..B + head?
src/net.*          sockets and the framed wire protocol
src/worker.cpp     the worker process
scripts/           weight export, Python client, local launcher
tests/             correctness check against PyTorch
docs/              per-part notes: concepts used and decisions taken
```
