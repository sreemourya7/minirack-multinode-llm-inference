# Part 5 notes: concepts and decisions

Part 4 gave the rack memory: every rank holds part of every session's cache.
That made generation fast, and it made a dead rank a real loss. Part 5
answers: **when a rank dies or freezes mid-generation, how does the rack
carry on without the client noticing more than a pause, and without the
answer changing by a single bit?**

```
 rank 1 dies       root notices: socket      replacement:         cache rebuilt:        the interrupted
 or hangs     -->  closed, or silent    -->  a waiting standby,   replay the tokens, --> step runs again
                   past the timeout          or a new process     or copy a checkpoint
```

Three separate problems hide in that picture: **noticing**, **replacing**,
and **rebuilding**. Each has a cheap answer and a fast answer, and the
measurements below show what each one buys.

## Concepts

**1. Crash versus hang.** A crashed process closes its sockets, so the root
sees the failure the moment it touches that socket. A hung process (frozen,
or cut off by the network) keeps its socket open and simply says nothing.
The two look completely different from the outside.

**2. Timeout.** The only way to notice a hang: decide how long silence is
acceptable. Too short and a rank that is merely slow gets declared dead. Too
long and a real hang stalls the client for that long. There is no right
value, only a trade-off. Code: `set_io_timeout_ms` in `src/net.cpp`.

**3. Fencing.** Once the root declares a rank dead, it closes that rank's
socket. If the old process was only frozen and wakes up later, it finds its
connection gone and exits. Without this, a "dead" rank could come back with
stale state and corrupt the computation.

**4. Commit point.** A request counts as done only when it has finished on
*every* rank. Only then does the root record it (the token in the history,
the rows in the checkpoint). So after any failure there is a well-defined
last good state to return to. Code: the "Commit" block in `Root::run_once`.

**5. Abort and resynchronise.** When one rank fails mid-step, the survivors
are in the middle of that step: computing, sending, or waiting. The root
sends each an ABORT, then reads and discards everything on that socket until
the rank's ABORT_ACK arrives. After the ack, the socket is known to be empty
in both directions. Code: `Mesh::abort_survivors`.

**6. Roll back, then redo.** Survivors may have appended half a step to
their caches. The root returns everyone to the last commit point, then runs
the interrupted request again from the start.

**7. Replay (recompute the state).** The root keeps each session's tokens,
4 bytes each. To rebuild, it re-runs them as one prefill. Costs nothing
until a failure. Rebuild time grows with the session.

**8. Checkpoint (copy the state).** After every step each rank sends the
root a copy of the cache rows it just added. To rebuild, the root sends them
back. Costs bytes and memory on every step. Rebuild is a memory copy.

**9. Why replay is exact.** Part 4 proved that prefill and step-by-step
decode build bit-identical caches. So re-running the tokens recreates
exactly the cache that was lost, and generation continues with the same
numbers. An earlier correctness property became a recovery mechanism.

**10. Why a rank cannot rebuild alone.** A rank's keys and values in layer 5
depend on the hidden state entering layer 5, which depends on *every* rank's
work in layers 0 to 4 (through the all-reduces). So a replacement cannot
recompute its cache by itself. Either the whole rack replays, or someone
kept a copy.

**11. Cold replacement versus standby.** A new process must first load its
slice of the weights (270 to 300 ms here, start-up included). A standby is a spare process that
has already done that and is waiting. Memory traded for recovery time.

**12. Supervisor.** The root can repair the rack, but something outside has
to start the replacement process. Here that is `scripts/supervisor.py`. In a
real cluster it is the cluster's process manager.

**13. Event loop.** The root now watches all of its sockets at once with
`poll`: new connections, the client, and the other ranks. A rank's socket is
silent between requests, so if it becomes "readable" the rank has gone, and
the root can repair before the next request even arrives.

**14. Degraded mode.** If no replacement comes within the wait, the client
gets a clear error and nothing is half-applied. When a replacement does
arrive, the session carries on from where it stopped.

**15. Single point of failure.** Everything above protects against losing a
non-root rank. The root itself holds the client connections, the token
history and the checkpoints. If it dies, the rack is gone.

## Decisions

| Decision | Why | What was given up |
|----------|-----|-------------------|
| Recovery is transparent: the root retries, the client just waits | The client needs no retry logic and cannot see a half-applied step | The client cannot choose to give up early |
| A request is committed only after every rank finishes it | Gives one clear state to roll back to | Nothing; it is bookkeeping |
| Abort with an acknowledgement, discarding everything before the ack | Leaves every surviving socket provably clean | One round trip per survivor |
| After a failure, every non-root rank starts from empty | One code path whether a rank is a survivor with half a step or a brand-new process | Survivors' intact caches are thrown away and re-sent or recomputed |
| Two strategies, chosen by a flag | They sit at opposite ends of a real trade-off, and both are measured | More code than one strategy |
| Checkpoints are kept on the root | The root already talks to every rank; no new connections | The root's memory grows, and the root is still a single point of failure |
| Checkpoint every step, not every N steps | Recovery never has to recompute anything | 24% more bytes on every step |
| Standbys identify themselves the same way as replacements | No new protocol: a HELLO for a healthy rank is a standby | One spare can only stand in for one specific rank |
| Close the dead rank's socket (fencing) | A woken-up rank can never rejoin with stale state | A healthy-but-slow rank is killed if it trips the timeout |
| One timeout for all steps | Simple | Must be set above the slowest legitimate step, so short hangs take that long to notice |
| No heartbeats while idle | Keeps ranks silent between requests, which is what lets the event loop treat any activity as a failure | A rank that freezes while idle is only noticed on the next request |
| A new connection is not read until it speaks | A silent connection cannot block the root | A few more lines in the event loop |
| The replica check failing still stops the rack | With two copies that differ there is no telling which is right | No recovery from silent corruption |

## Measured (4-core laptop, Intel Core i5-10300H; 2 ranks, AVX2 kernels)

The step that hit the failure, 64-token session (a normal step is 38 to 40 ms):

| Replacement | State rebuilt by | That step | Waiting for a rank | Rebuilding |
|-------------|------------------|-----------|--------------------|------------|
| new process | replay | 697 ms | 299 ms | 357 ms |
| new process | checkpoint | 316 ms | 268 ms | 1.6 ms |
| standby | replay | 354 ms | 0 ms | 316 ms |
| standby | checkpoint | 42 ms | 0 ms | 1.3 ms |

Rebuild time by session length:

| Tokens | Replay | Checkpoint |
|--------|--------|------------|
| 16 | 78 ms | 0.6 ms |
| 64 | 271 ms | 1.7 ms |
| 128 | 494 ms | 5.8 ms |
| 256 | 979 ms | 6.7 ms |
| 512 | 2186 ms | 21.7 ms |

What checkpointing costs when nothing fails: 188 kB per decode step instead
of 151 kB (24% more), and 37 kB of root memory per token. Step time was 38.7
ms with replay and 38.6 ms with checkpoint, so no time cost was measurable.

A frozen rank with a 1000 ms timeout: that step took 1213 ms.

Replay is a prefill, so its speed follows the kernels. These numbers are with
the AVX2 kernels of Part 6; with the scalar kernels a prefill is about 9
times slower on this machine, and replay would be too.

## Summary

`kill -9` a rank in the middle of generating text. With a standby and
checkpointing, the client sees that token take 42 ms instead of about 40,
gets no error, and the output is bit-identical to a run with no failure. If
the kill lands in the middle of a step, that step is run again, so it costs
about one extra step. Without a standby or checkpoints the same failure takes
0.7 seconds, and replay gets worse with context length: 2.2 seconds at 512
tokens against 22 milliseconds for checkpoint. The price of checkpointing is
24% more bytes per step and a second copy of the cache.

The points underneath that result:
- The three sub-problems (notice, replace, rebuild) and the fast and cheap
  answer to each.
- Why a hang is harder than a crash, and why the timeout is a trade-off.
- Why replay is exact (the Part 4 property), and why a rank cannot rebuild
  its cache alone.
- What is still unprotected: the root.

## Questions worth being able to answer

- How does the root tell a crashed rank from a frozen one?
- What could go wrong if the root did not close a "dead" rank's socket?
- A rank fails halfway through a decode step. What state is each surviving
  rank in, and how is it brought back in step?
- Why can the replacement rank not rebuild its own cache?
- When would you choose replay over checkpoint?
- What would it take to survive the loss of the root?
- The timeout is 10 s by default. What is the argument for making it shorter,
  and what breaks if you do?
