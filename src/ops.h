// ops.h - the handful of math operations a GPT-2 decoder block needs.
//
// Part 1 keeps these as plain scalar loops. They are the correctness baseline;
// Part 5 replaces the hot ones with AVX2 and measures the difference.
#pragma once

#include "mat.h"

namespace mr {

// y = x @ w + b
//   x: [T, in]   w: [in, out]   b: [1, out]   ->   y: [T, out]
// GPT-2 stores its weights as [in, out] (its "Conv1D" layer), which is the
// transpose of torch.nn.Linear. We keep the file layout and index accordingly.
Mat linear(const Mat& x, const Mat& w, const Mat& b);

// Layer normalisation over the feature axis, one row (token) at a time.
Mat layernorm(const Mat& x, const Mat& gamma, const Mat& beta, float eps);

// GPT-2's GELU (the tanh approximation, "gelu_new" in Hugging Face).
void gelu_inplace(Mat& x);

// a += b, elementwise. Used for the residual connections.
void add_inplace(Mat& a, const Mat& b);

// Causal multi-head self-attention.
//   qkv: [T, 3*D], the output of the c_attn linear layer, laid out as
//        [query | key | value] along the feature axis.
//   returns [T, D] (before the output projection).
// "Causal" means token i may only look at tokens 0..i.
Mat causal_self_attention(const Mat& qkv, int n_head);

}  // namespace mr
