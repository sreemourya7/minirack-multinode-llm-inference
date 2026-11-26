// tp_model.h - tensor parallelism: GPT-2 cut INSIDE each layer.
//
// Part 1 cut the model between layers, so the workers took turns.
// Here every rank works on every layer at the same time. Each big weight
// matrix is sliced, and rank r owns slice r:
//
//   attention   12 heads           -> rank r owns a contiguous group of heads
//   MLP         3072 hidden units  -> rank r owns a contiguous group of units
//   head        50257 vocabulary   -> rank r owns a contiguous range of tokens
//
// Why this is exact, not an approximation: each of those layers ends in a
// SUM over heads / hidden units. A sum can be split into partial sums. Each
// rank computes its partial sum, and all_reduce_sum adds the parts together.
//
// The small things (layernorm weights, biases) are simply copied to every
// rank. Every rank also keeps its own full copy of the hidden state h.
#pragma once

#include <cstdint>
#include <vector>

#include "mat.h"
#include "mesh.h"
#include "weights.h"

namespace mr {

struct TpBlock {
  Mat ln1_g, ln1_b;      // replicated
  Mat qkv_w, qkv_b;      // [D, 3L], [1, 3L]: query|key|value columns of MY heads (L = my heads x head size)
  Mat aproj_w;           // [L, D]: rows of the attention output projection for MY heads
  Mat aproj_b;           // [1, D]: replicated, added once after the all-reduce
  Mat ln2_g, ln2_b;      // replicated
  Mat fc_w, fc_b;        // [D, U], [1, U]: MLP-up columns for MY hidden units
  Mat mproj_w;           // [U, D]: MLP-down rows for MY hidden units
  Mat mproj_b;           // [1, D]: replicated, added once after the all-reduce
};

struct TpShard {
  ModelConfig cfg;
  int rank = 0;
  int nranks = 1;
  int head_begin = 0, head_end = 0;    // my attention heads
  int unit_begin = 0, unit_end = 0;    // my MLP hidden units
  int vocab_begin = 0, vocab_end = 0;  // my slice of the vocabulary

  std::vector<TpBlock> blocks;
  Mat lnf_g, lnf_b;  // replicated
  // Root: the full token table (it needs every row to embed the input); its
  // own vocabulary slice is the first rows. Other ranks: just their slice.
  Mat wte;
  Mat wpe;           // root only

  uint64_t resident_bytes = 0;  // weights this rank keeps in memory
};

// Slice [0, n) into `parts` contiguous pieces of near-equal size.
inline int part_begin(int n, int parts, int i) { return static_cast<int>(static_cast<int64_t>(n) * i / parts); }

TpShard load_tp_shard(WeightFile& wf, int rank, int nranks);

// Runs all layers on h. Every rank calls this with the same h and ends with
// the same h. Two all-reduces per layer.
void tp_forward(const TpShard& s, Mesh& mesh, Mat& h);

// Final layernorm, then logits for MY slice of the vocabulary only.
Mat tp_head_local(const TpShard& s, const Mat& h, bool last_only);

}  // namespace mr
