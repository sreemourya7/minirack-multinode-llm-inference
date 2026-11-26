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
slice of the weights (about 210 ms here). A standby is a spare process that
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

## Measured (2-core development machine, 2 ranks)

The step that hit the failure, 64-token session (a normal step is 55 to 61 ms):

| Replacement | State rebuilt by | That step | Waiting for a rank | Rebuilding |
|-------------|------------------|-----------|--------------------|------------|
| new process | replay | 1635 ms | 208 ms | 1368 ms |
| new process | checkpoint | 274 ms | 211 ms | 1.8 ms |
| standby | replay | 1333 ms | 0 ms | 1279 ms |
| standby | checkpoint | 76 ms | 0 ms | 2.0 ms |

Rebuild time by session length:

| Tokens | Replay | Checkpoint |
|--------|--------|------------|
| 16 | 348 ms | 0.2 ms |
| 64 | 1341 ms | 2.0 ms |
| 128 | 2646 ms | 6.3 ms |
| 256 | 5763 ms | 11.7 ms |
| 512 | 11013 ms | 31.0 ms |

What checkpointing costs when nothing fails: 188 kB per decode step instead
of 151 kB (24% more), and 37 kB of root memory per token. Step time was 57 to
61 ms with either strategy, so the time cost is below the noise on loopback.

A frozen rank with a 1000 ms timeout: that step took about 1070 ms.

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
