# Documentation

Start with the [project README](../README.md) for what minirack is and what it
achieves. This folder holds the detail.

| Document | Read it for |
|----------|-------------|
| [ARCHITECTURE.md](ARCHITECTURE.md) | How the system is put together: processes, wire protocol, the path of a request, state, failure handling |
| [RESULTS.md](RESULTS.md) | Every check and measurement, part by part |

## Notes, part by part

The project was built in six parts, each adding one capability on top of the
last. Each set of notes has the same shape: the question the part answers,
the **concepts** it uses (explained from scratch), the **decisions** taken and
what each one cost, the **measurements**, and a list of **questions** to test
understanding.

| Part | Question it answers | Main concepts |
|------|---------------------|---------------|
| [1. Pipeline split](part1-notes.md) | Can a model be cut into pieces in different processes and still give the right answer? | The GPT-2 forward pass, sharded weight loading, message framing, a golden reference |
| [2. Tensor parallelism](part2-notes.md) | How can several nodes work on the same layer at the same time? | SPMD, collectives, all-reduce, replicated state, floating-point summation order |
| [3. Mixture of Experts](part3-notes.md) | How does a runtime execute a model that uses only part of itself per token? | Routers, top-k, gates, sparse gather and scatter, load imbalance, anchor tests |
| [4. KV cache](part4-notes.md) | How does generation avoid redoing all earlier work for every new token? | Prefill and decode, sessions, sharded cache, exactness, admission limits |
| [5. Failure recovery](part5-notes.md) | How does the rack survive a node that dies or hangs mid-generation? | Failure detection, fencing, commit points, abort and resync, replay against checkpoint, standbys |
| [6. Kernels and profile](part6-notes.md) | Where does the time go, and what is the hardware limit? | Profiling, SIMD and FMA, cache behaviour, run-time dispatch, memory bandwidth, rooflines |

## Charts and data

| File | Contents |
|------|----------|
| [part6-profile.svg](part6-profile.svg) | Time per token by operation, before and after the AVX2 kernels |
| [part5-recovery.svg](part5-recovery.svg) | Time to rebuild a lost rank's cache: replay against checkpoint |
| [data/](data/) | Raw output of the runs behind every number, and the CSV files behind the two charts |

All numbers come from one machine: a 4-core, 8-thread Intel Core i5-10300H
laptop running Ubuntu 24.04 under WSL2, g++ 13.3. To reproduce them, follow
the quick start in the project README; `scripts/plot_profile.py` and
`scripts/plot_recovery.py` redraw the charts from a fresh run.
