# Part 2 notes: concepts and decisions

Part 1 showed a model can be cut into pieces. But its pieces took turns, so
one request got no faster. Part 2 answers: **can the pieces work at the same
time?** Yes, if the cut goes *inside* each layer and the ranks combine their
partial results with a collective.

```
                  +--> rank 0: heads 0-5,  MLP units 0-1535    --+
 hidden state h --+                                              +--> all-reduce (sum) --> next layer
 (same on all)    +--> rank 1: heads 6-11, MLP units 1536-3071 --+
```

## Concepts

**1. Rank.** One process in a group that works together. Ranks are numbered
0 to N-1. Rank 0 is the root: it also talks to the client.

**2. Same program, different data (SPMD).** Every rank runs the *same* code
(`run_request` in `src/rank.cpp`). What differs is the slice of weights each
one loaded. Part 1 was different: each worker ran a different stage.

**3. Tensor parallelism.** Slice each big weight matrix across ranks:
12 attention heads, 3072 MLP hidden units, 50257 vocabulary rows. Rank r owns
slice r of each. Code: `load_tp_shard` in `src/tp_model.cpp`.

**4. Why slicing is exact.** Each layer ends in a sum: over heads in
attention, over hidden units in the MLP. A sum can be done in parts:
`(a + b) + (c + d)` is the same total as `a + b + c + d`. So each rank computes
its partial sum and the parts are added. Nothing is approximated.

**5. Collective.** A step all ranks do together: everyone stops, data is
exchanged, everyone continues with a shared result. A request here has 27 of
them: 1 broadcast, 24 all-reduces (2 per layer), 1 replica check, 1 gather.

**6. All-reduce.** Every rank puts in a matrix; every rank gets back the sum.
Here it is done as "reduce to the root, then broadcast": the others send
their part to rank 0, rank 0 adds them, rank 0 sends the total back.
Code: `Mesh::all_reduce_sum` in `src/mesh.cpp`.

**7. Gather.** Every rank puts in a piece; the root gets all the pieces side
by side. Used for the logits: each rank scores its own slice of the
vocabulary.

**8. Replicated state.** Every rank keeps its own full copy of the hidden
state `h`. The copies must stay bit-identical, or the ranks would be computing
on different inputs without knowing.

**9. Deterministic reduction order.** Float addition depends on order (Part 1,
concept 10). So the sum is formed in one place (the root), in one fixed order
(rank 0 + 1 + 2 ...), and the *result* is what gets shared. If each rank added
the parts in its own order, the copies of `h` could differ in the last bit and
drift apart layer by layer.

**10. Replica check.** At the end of every request each rank sends a checksum
of its `h` to the root, which verifies they all match. The test suite forces
one rank to corrupt its copy and confirms the check catches it.

**11. Sequence numbers.** Every message inside a request carries
`(request_id, seq)`, where `seq` counts collectives. If two ranks ever
disagree about which step they are on, the receiver sees a wrong `seq` and
stops, instead of adding the wrong matrices together.

**12. Bias once.** A bias must be added one time, not once per rank. So each
rank computes its part with no bias (`matmul`), and the bias is added after
the all-reduce.

**13. Oversubscription.** More ranks than CPU cores means ranks share cores.
On the 2-core test machine, 2 ranks gave 1.96x and 4 ranks gave no more.

## Decisions

| Decision | Why | What was given up |
|----------|-----|-------------------|
| Star wiring: every rank connects only to the root | One connection per rank; no deadlock possible with blocking sockets, because the order of sends and receives is fixed | Everything flows through the root, which becomes the bottleneck as ranks grow |
| All-reduce as reduce-then-broadcast | Simple, and gives the fixed summation order for free | Two hops per all-reduce; a ring or tree would spread the load |
| Replicate the hidden state on every rank | Each rank can start the next layer immediately, with no extra exchange | Memory for N copies of `h` (small: 98 KB for 32 tokens) |
| Replicate small weights (layernorm, biases) | They are tiny; slicing them would add collectives | A few KB per rank |
| Root embeds, then broadcasts `h` | Only one rank needs the full 154 MB token table for lookup | The root's shard is bigger than the others |
| Slice the vocabulary for the head | The head is a large matmul; splitting it lets it run in parallel too | One gather at the end |
| Validate the request on the root before the broadcast | A bad request never reaches the other ranks, so they cannot fall out of step | Nothing |
| Replica check on every request | Catches silent divergence at once | 8 bytes per rank and one checksum pass over `h` |
| After any failure mid-request, mark the rack down | The ranks are no longer in step; guessing would risk wrong answers | Availability, until Part 5 adds recovery |
| A fault-injection hook (`MR_TEST_CORRUPT_RANK`) | A check that has never been seen to fail is not verified | A few lines of test-only code in the binary |
| Keep Part 1's pipeline worker as a separate binary | Each stays readable; the two splits are different ideas | Some duplicated setup code |

## Measured (2-core development machine, 32-token request)

| Ranks | Latency | Speedup | In collectives (root) | Data through the root |
|-------|---------|---------|-----------------------|-----------------------|
| 1 | 1168 ms | 1.00x | 0 ms | 0 MB |
| 2 | 596 ms | 1.96x | 13 ms | 4.9 MB |
| 4 | 596 ms | 1.96x | 50 ms | 14.6 MB |

Where the 4.9 MB comes from: each all-reduce moves one `[32, 768]` matrix to
the root and one back, 98 KB each way. 24 all-reduces x 2 x 98 KB = 4.7 MB,
plus the broadcast and the gather.

## Questions worth being able to answer

- Why does tensor parallelism speed up one request when a pipeline split
  does not?
- Why is splitting the MLP across ranks exact rather than approximate?
- What goes wrong if each rank sums the partial results in its own order?
- How many collectives does one request need, and what are they?
- Data through the root grows with the number of ranks. Why, and what wiring
  would avoid it?
- On this machine communication is about 2% of the time. What would make it a
  much larger share? (Hint: think about very short requests, such as decoding
  one token at a time, and about real networks instead of loopback.)
