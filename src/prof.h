// prof.h - where does the time go? (Part 6)
//
// Each of the model's basic operations adds the time it took to a counter.
// A rank resets the counters at the start of a request and prints them at
// the end. Measure first, optimise second: the counters say which loop is
// worth rewriting and, afterwards, what the rewrite actually bought.
#pragma once

#include <chrono>

namespace mr::prof {

enum Phase {
  kMatmul = 0,  // every linear layer: attention's projections, the MLP, the experts
  kAttention,   // scores, softmax, and the weighted sum of values
  kGelu,
  kNorm,        // layer normalisation
  kHead,        // the vocabulary projection
  kCount
};

extern double ms[kCount];  // one process, one thread: plain globals are enough
void reset();

// Adds the lifetime of this object to one counter.
struct Scope {
  Phase phase;
  std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
  explicit Scope(Phase p) : phase(p) {}
  ~Scope() { ms[phase] += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count(); }
};

}  // namespace mr::prof
