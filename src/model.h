// model.h - a "shard" of GPT-2: the slice of the model one worker owns.
//
// The whole model is:   tokens -> embed -> block 0 .. block N-1 -> head -> logits
// A shard owns a contiguous range of blocks, plus optionally the embedding
// (first worker) and/or the head (last worker). Chaining shards end to end
// reproduces the full model.
#pragma once

#include <cstdint>
#include <vector>

#include "mat.h"
#include "weights.h"

namespace mr {

struct BlockWeights {
  Mat ln1_g, ln1_b;        // layernorm before attention
  Mat attn_w, attn_b;      // c_attn: D -> 3D  (query | key | value)
  Mat aproj_w, aproj_b;    // attention output projection: D -> D
  Mat ln2_g, ln2_b;        // layernorm before the MLP
  Mat fc_w, fc_b;          // MLP up:   D -> 4D
  Mat mproj_w, mproj_b;    // MLP down: 4D -> D
};

struct Shard {
  ModelConfig cfg;
  int first_layer = 0;  // inclusive
  int last_layer = 0;   // exclusive
  bool has_embed = false;
  bool has_head = false;

  std::vector<BlockWeights> blocks;  // blocks[i] is layer first_layer + i
  Mat wte;                           // token embedding [vocab, D]; also the head's output matrix
  Mat wpe;                           // position embedding [n_ctx, D]
  Mat lnf_g, lnf_b;                  // final layernorm
};

// Loads only the tensors this shard needs.
Shard load_shard(WeightFile& wf, int first_layer, int last_layer, bool has_embed, bool has_head);

// tokens -> [T, D] hidden states (token embedding + position embedding).
Mat embed(const Shard& s, const std::vector<int32_t>& ids);

// The same lookup, given the two tables directly (used by the Part 2 ranks).
// pos0 is the position of the first token: 0 for a fresh sequence, or the
// current length of a session when decoding one more token (Part 4).
Mat embed_tokens(const ModelConfig& cfg, const Mat& wte, const Mat& wpe, const std::vector<int32_t>& ids,
                 int pos0 = 0);

// Runs this shard's blocks in place on h ([T, D]).
void run_blocks(const Shard& s, Mat& h);

// Final layernorm + projection onto the vocabulary.
// last_only = true returns logits for the final token only ([1, vocab]),
// which is all that text generation needs.
Mat lm_head(const Shard& s, const Mat& h, bool last_only);

}  // namespace mr
