// moe.h - Mixture of Experts (Part 3).
//
// A dense MLP runs ALL of its hidden units for EVERY token. An MoE layer
// splits the MLP into E smaller MLPs ("experts") and, for each token, runs
// only the best k of them:
//
//   1. router:  scores = x @ router_w + router_b        one score per expert
//   2. top-k:   keep the k highest-scoring experts for this token
//   3. gates:   softmax over those k scores              how much to trust each
//   4. output:  sum over the k chosen experts of  gate * expert(x)   (+ shared bias)
//
// Here expert e is a contiguous group of GPT-2's MLP hidden units, so with
// every expert switched on and every gate equal to 1 the layer is exactly the
// original MLP. That gives the tests a known-correct anchor.
//
// Across ranks: rank r owns a contiguous range of experts. Every rank computes
// the routing itself (same input, same code, same answer), runs only its own
// experts on only the tokens routed to them, and the all-reduce from Part 2
// adds the pieces together.
#pragma once

#include <cstdint>
#include <vector>

#include "mat.h"

namespace mr {

struct MoeConfig {
  bool enabled = false;
  int n_expert = 0;
  int top_k = 0;
  bool softmax_gate = true;  // false: every chosen expert gets gate 1 (used by the anchor test)
};

// One expert: D -> units -> D. No output bias of its own; the layer adds one
// shared bias after the experts are summed.
struct Expert {
  Mat fc_w;     // [D, units]
  Mat fc_b;     // [1, units]
  Mat mproj_w;  // [units, D]
};

// The router's decision for T tokens: for token t, entries t*k .. t*k+k-1.
struct Routing {
  int T = 0;
  int k = 0;
  std::vector<int32_t> expert;  // chosen expert ids
  std::vector<float> gate;      // their weights
};

// Steps 1-3 above. Ties go to the lower expert id, so the result is fully
// determined by the bits of x.
Routing moe_route(const Mat& x, const Mat& router_w, const Mat& router_b, const MoeConfig& cfg);

// Step 4, for the experts this rank owns (ids expert_begin, expert_begin+1, ...).
// Each expert runs only on the tokens routed to it. Returns this rank's share
// of the layer output, [T, D]; rows with no local expert stay zero.
// *rows_run is increased by the number of (token, expert) pairs executed.
Mat moe_run_local(const Mat& x, const Routing& route, const std::vector<Expert>& experts, int expert_begin,
                  uint64_t* rows_run);

}  // namespace mr
