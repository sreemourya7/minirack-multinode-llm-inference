// kv_cache.h - per-request state: the KV cache (Part 4).
//
// The problem. To produce token N+1 the model needs attention over tokens
// 0..N. Without a cache, every new token re-runs the whole sequence, so
// generating gets slower and slower as the text grows.
//
// The observation. In attention, each token contributes a key and a value
// per layer, and causality means an old token's key and value never change
// when new tokens arrive. So compute them once and keep them.
//
//   prefill  run the whole prompt once; store every token's key and value
//   decode   run ONE new token: compute its key and value, append them, and
//            let it attend over everything stored so far
//
// A session is the stored state of one conversation: its length so far and,
// for every layer, the keys and values of every token.
//
// Across ranks: each rank stores keys and values only for the attention heads
// it owns. The cache is sharded exactly like the weights, and it never
// crosses the wire.
#pragma once

#include <cstdint>
#include <unordered_map>
#include <vector>

#include "mat.h"

namespace mr {

// One layer's cache: `len` rows of `width` floats each, for keys and values.
struct KvLayer {
  std::vector<float> k;
  std::vector<float> v;
};

struct Session {
  int len = 0;    // tokens stored so far
  int width = 0;  // floats per token per layer = my heads x head size
  std::vector<KvLayer> layers;

  uint64_t bytes() const { return static_cast<uint64_t>(len) * width * sizeof(float) * 2 * layers.size(); }
};

// All sessions on this rank, by id. Every rank creates and erases the same
// sessions in the same order, so the tables stay in step.
class SessionTable {
 public:
  static constexpr size_t kMaxSessions = 64;

  Session& create(uint64_t id, int n_layer, int width);  // throws if the id exists or the table is full
  Session* find(uint64_t id);                            // nullptr if absent
  bool erase(uint64_t id);
  size_t size() const { return sessions_.size(); }

 private:
  std::unordered_map<uint64_t, Session> sessions_;
};

// Attention with a cache.
//   qkv:     [Tn, 3L] for the Tn NEW tokens only (query | key | value)
//   cache:   holds old_len tokens on entry, old_len + Tn on return
// Appends the new keys and values, then lets each new token attend to every
// stored token up to and including itself. Returns [Tn, L].
//
// With an empty cache and the whole sequence as "new" this does exactly the
// arithmetic of causal_self_attention, in the same order. That is what makes
// cached generation bit-identical to recomputing from scratch.
Mat attend_cached(const Mat& qkv, int n_head, KvLayer& cache, int old_len);

}  // namespace mr
