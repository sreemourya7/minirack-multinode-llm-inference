#include "tp_model.h"

#include <stdexcept>
#include <string>

#include "ops.h"

namespace mr {

namespace {

// Copies columns [c0, c1) of m.
Mat slice_cols(const Mat& m, int c0, int c1) {
  if (c0 < 0 || c1 > m.cols || c0 > c1) throw std::runtime_error("slice_cols: bad range");
  Mat out(m.rows, c1 - c0);
  for (int r = 0; r < m.rows; ++r) {
    const float* src = m.row(r) + c0;
    float* dst = out.row(r);
    for (int c = 0; c < out.cols; ++c) dst[c] = src[c];
  }
  return out;
}

// Puts three matrices side by side.
Mat hcat3(const Mat& a, const Mat& b, const Mat& c) {
  Mat out(a.rows, a.cols + b.cols + c.cols);
  for (int r = 0; r < a.rows; ++r) {
    float* dst = out.row(r);
    for (int i = 0; i < a.cols; ++i) *dst++ = a.row(r)[i];
    for (int i = 0; i < b.cols; ++i) *dst++ = b.row(r)[i];
    for (int i = 0; i < c.cols; ++i) *dst++ = c.row(r)[i];
  }
  return out;
}

}  // namespace

TpShard load_tp_shard(WeightFile& wf, int rank, int nranks) {
  TpShard s;
  s.cfg = wf.config();
  s.rank = rank;
  s.nranks = nranks;
  const int D = static_cast<int>(s.cfg.n_embd);
  const int H = static_cast<int>(s.cfg.n_head);
  const int V = static_cast<int>(s.cfg.vocab_size);
  const int dh = D / H;
  if (nranks > H) throw std::runtime_error("more ranks than attention heads: each rank needs at least one head");

  s.head_begin = part_begin(H, nranks, rank);
  s.head_end = part_begin(H, nranks, rank + 1);
  s.vocab_begin = part_begin(V, nranks, rank);
  s.vocab_end = part_begin(V, nranks, rank + 1);
  const int a0 = s.head_begin * dh, a1 = s.head_end * dh;  // my columns inside one of q / k / v

  for (int l = 0; l < static_cast<int>(s.cfg.n_layer); ++l) {
    const std::string p = "h." + std::to_string(l) + ".";
    TpBlock b;
    b.ln1_g = wf.get(p + "ln_1.weight");
    b.ln1_b = wf.get(p + "ln_1.bias");
    {
      // c_attn maps D -> [query | key | value], each D wide. My heads are the
      // same column range inside each of the three sections.
      const Mat w = wf.get(p + "attn.c_attn.weight");
      const Mat bias = wf.get(p + "attn.c_attn.bias");
      b.qkv_w = hcat3(slice_cols(w, a0, a1), slice_cols(w, D + a0, D + a1), slice_cols(w, 2 * D + a0, 2 * D + a1));
      b.qkv_b = hcat3(slice_cols(bias, a0, a1), slice_cols(bias, D + a0, D + a1), slice_cols(bias, 2 * D + a0, 2 * D + a1));
    }
    b.aproj_w = wf.get_rows(p + "attn.c_proj.weight", a0, a1);
    b.aproj_b = wf.get(p + "attn.c_proj.bias");
    b.ln2_g = wf.get(p + "ln_2.weight");
    b.ln2_b = wf.get(p + "ln_2.bias");
    {
      const Mat w = wf.get(p + "mlp.c_fc.weight");
      const Mat bias = wf.get(p + "mlp.c_fc.bias");
      s.unit_begin = part_begin(w.cols, nranks, rank);
      s.unit_end = part_begin(w.cols, nranks, rank + 1);
      b.fc_w = slice_cols(w, s.unit_begin, s.unit_end);
      b.fc_b = slice_cols(bias, s.unit_begin, s.unit_end);
    }
    b.mproj_w = wf.get_rows(p + "mlp.c_proj.weight", s.unit_begin, s.unit_end);
    b.mproj_b = wf.get(p + "mlp.c_proj.bias");

    for (const Mat* m : {&b.ln1_g, &b.ln1_b, &b.qkv_w, &b.qkv_b, &b.aproj_w, &b.aproj_b, &b.ln2_g, &b.ln2_b, &b.fc_w,
                         &b.fc_b, &b.mproj_w, &b.mproj_b}) {
      s.resident_bytes += m->size() * sizeof(float);
    }
    s.blocks.push_back(std::move(b));
  }

  s.lnf_g = wf.get("ln_f.weight");
  s.lnf_b = wf.get("ln_f.bias");
  if (rank == 0) {
    s.wte = wf.get("wte.weight");  // vocab_begin is 0 here, so my slice is rows [0, vocab_end)
    s.wpe = wf.get("wpe.weight");
  } else {
    s.wte = wf.get_rows("wte.weight", s.vocab_begin, s.vocab_end);
  }
  s.resident_bytes += (s.lnf_g.size() + s.lnf_b.size() + s.wte.size() + s.wpe.size()) * sizeof(float);
  return s;
}

void tp_forward(const TpShard& s, Mesh& mesh, Mat& h) {
  const int my_heads = s.head_end - s.head_begin;
  for (const TpBlock& b : s.blocks) {
    // --- attention half ---
    {
      Mat x = layernorm(h, b.ln1_g, b.ln1_b, 1e-5f);     // same on every rank
      Mat qkv = linear(x, b.qkv_w, b.qkv_b);             // q, k, v for MY heads only
      Mat a = causal_self_attention(qkv, my_heads);      // [T, L]
      Mat y = matmul(a, b.aproj_w);                      // MY heads' share of the output, [T, D]
      mesh.all_reduce_sum(y);                            // now y is the sum over ALL heads
      add_row_inplace(y, b.aproj_b);
      add_inplace(h, y);
    }
    // --- MLP half ---
    {
      Mat x = layernorm(h, b.ln2_g, b.ln2_b, 1e-5f);
      Mat up = linear(x, b.fc_w, b.fc_b);                // MY hidden units only, [T, U]
      gelu_inplace(up);
      Mat y = matmul(up, b.mproj_w);                     // MY units' share of the output, [T, D]
      mesh.all_reduce_sum(y);                            // sum over ALL hidden units
      add_row_inplace(y, b.mproj_b);
      add_inplace(h, y);
    }
  }
}

Mat tp_head_local(const TpShard& s, const Mat& h, bool last_only) {
  const Mat x = layernorm(h, s.lnf_g, s.lnf_b, 1e-5f);
  const int D = x.cols;
  const int n_local = s.vocab_end - s.vocab_begin;
  const int first = last_only ? x.rows - 1 : 0;
  Mat logits(x.rows - first, n_local);
  for (int t = first; t < x.rows; ++t) {
    const float* xt = x.row(t);
    float* lt = logits.row(t - first);
    for (int v = 0; v < n_local; ++v) {
      const float* wv = s.wte.row(v);  // row v of my slice (on the root the slice starts at row 0)
      float acc = 0.0f;
      for (int d = 0; d < D; ++d) acc += xt[d] * wv[d];
      lt[v] = acc;
    }
  }
  return logits;
}

}  // namespace mr
