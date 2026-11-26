#include "model.h"

#include <stdexcept>
#include <string>

#include "ops.h"

namespace mr {

Shard load_shard(WeightFile& wf, int first_layer, int last_layer, bool has_embed, bool has_head) {
  Shard s;
  s.cfg = wf.config();
  if (first_layer < 0 || last_layer > static_cast<int>(s.cfg.n_layer) || first_layer > last_layer) {
    throw std::runtime_error("bad layer range for this model");
  }
  s.first_layer = first_layer;
  s.last_layer = last_layer;
  s.has_embed = has_embed;
  s.has_head = has_head;

  for (int l = first_layer; l < last_layer; ++l) {
    const std::string p = "h." + std::to_string(l) + ".";
    BlockWeights b;
    b.ln1_g = wf.get(p + "ln_1.weight");
    b.ln1_b = wf.get(p + "ln_1.bias");
    b.attn_w = wf.get(p + "attn.c_attn.weight");
    b.attn_b = wf.get(p + "attn.c_attn.bias");
    b.aproj_w = wf.get(p + "attn.c_proj.weight");
    b.aproj_b = wf.get(p + "attn.c_proj.bias");
    b.ln2_g = wf.get(p + "ln_2.weight");
    b.ln2_b = wf.get(p + "ln_2.bias");
    b.fc_w = wf.get(p + "mlp.c_fc.weight");
    b.fc_b = wf.get(p + "mlp.c_fc.bias");
    b.mproj_w = wf.get(p + "mlp.c_proj.weight");
    b.mproj_b = wf.get(p + "mlp.c_proj.bias");
    s.blocks.push_back(std::move(b));
  }

  // GPT-2 ties its weights: the same matrix (wte) turns tokens into vectors at
  // the start and vectors back into token scores at the end. So both the embed
  // worker and the head worker need a copy.
  if (has_embed || has_head) s.wte = wf.get("wte.weight");
  if (has_embed) s.wpe = wf.get("wpe.weight");
  if (has_head) {
    s.lnf_g = wf.get("ln_f.weight");
    s.lnf_b = wf.get("ln_f.bias");
  }
  return s;
}

Mat embed(const Shard& s, const std::vector<int32_t>& ids) {
  if (!s.has_embed) throw std::runtime_error("this worker does not hold the embedding");
  return embed_tokens(s.cfg, s.wte, s.wpe, ids);
}

Mat embed_tokens(const ModelConfig& cfg, const Mat& wte, const Mat& wpe, const std::vector<int32_t>& ids) {
  const int T = static_cast<int>(ids.size());
  const int D = static_cast<int>(cfg.n_embd);
  if (T == 0) throw std::runtime_error("empty token sequence");
  if (T > static_cast<int>(cfg.n_ctx)) throw std::runtime_error("sequence longer than the context window");
  Mat h(T, D);
  for (int t = 0; t < T; ++t) {
    const int32_t id = ids[t];
    if (id < 0 || id >= static_cast<int32_t>(cfg.vocab_size)) throw std::runtime_error("token id out of range");
    const float* te = wte.row(id);
    const float* pe = wpe.row(t);
    float* ht = h.row(t);
    for (int d = 0; d < D; ++d) ht[d] = te[d] + pe[d];
  }
  return h;
}

void run_blocks(const Shard& s, Mat& h) {
  if (h.cols != static_cast<int>(s.cfg.n_embd)) throw std::runtime_error("hidden state has the wrong width");
  for (const BlockWeights& b : s.blocks) {
    // --- attention half:  h = h + proj(attention(ln1(h))) ---
    {
      Mat x = layernorm(h, b.ln1_g, b.ln1_b, 1e-5f);
      Mat qkv = linear(x, b.attn_w, b.attn_b);
      Mat a = causal_self_attention(qkv, static_cast<int>(s.cfg.n_head));
      Mat y = linear(a, b.aproj_w, b.aproj_b);
      add_inplace(h, y);
    }
    // --- MLP half:  h = h + down(gelu(up(ln2(h)))) ---
    {
      Mat x = layernorm(h, b.ln2_g, b.ln2_b, 1e-5f);
      Mat up = linear(x, b.fc_w, b.fc_b);
      gelu_inplace(up);
      Mat y = linear(up, b.mproj_w, b.mproj_b);
      add_inplace(h, y);
    }
  }
}

Mat lm_head(const Shard& s, const Mat& h, bool last_only) {
  if (!s.has_head) throw std::runtime_error("this worker does not hold the head");
  const Mat x = layernorm(h, s.lnf_g, s.lnf_b, 1e-5f);
  const int D = x.cols;
  const int V = static_cast<int>(s.cfg.vocab_size);
  const int first = last_only ? x.rows - 1 : 0;
  Mat logits(x.rows - first, V);
  for (int t = first; t < x.rows; ++t) {
    const float* xt = x.row(t);
    float* lt = logits.row(t - first);
    for (int v = 0; v < V; ++v) {
      const float* wv = s.wte.row(v);
      float acc = 0.0f;
      for (int d = 0; d < D; ++d) acc += xt[d] * wv[d];
      lt[v] = acc;
    }
  }
  return logits;
}

}  // namespace mr
