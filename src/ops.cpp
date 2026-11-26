#include "ops.h"

#include <cmath>
#include <stdexcept>
#include <vector>

namespace mr {

Mat linear(const Mat& x, const Mat& w, const Mat& b) {
  if (x.cols != w.rows || b.cols != w.cols) throw std::runtime_error("linear: shape mismatch");
  const int T = x.rows, in = w.rows, out = w.cols;
  Mat y(T, out);
  for (int t = 0; t < T; ++t) {
    float* yt = y.row(t);
    const float* xt = x.row(t);
    for (int o = 0; o < out; ++o) yt[o] = b.data[o];
    // Loop order t, i, o: the inner loop walks one row of w and one row of y,
    // both contiguous in memory, so the cache (and the compiler) are happy.
    for (int i = 0; i < in; ++i) {
      const float xi = xt[i];
      const float* wi = w.row(i);
      for (int o = 0; o < out; ++o) yt[o] += xi * wi[o];
    }
  }
  return y;
}

Mat matmul(const Mat& x, const Mat& w) {
  if (x.cols != w.rows) throw std::runtime_error("matmul: shape mismatch");
  const int T = x.rows, in = w.rows, out = w.cols;
  Mat y(T, out);
  for (int t = 0; t < T; ++t) {
    float* yt = y.row(t);
    const float* xt = x.row(t);
    for (int i = 0; i < in; ++i) {
      const float xi = xt[i];
      const float* wi = w.row(i);
      for (int o = 0; o < out; ++o) yt[o] += xi * wi[o];
    }
  }
  return y;
}

void add_row_inplace(Mat& a, const Mat& row) {
  if (row.rows != 1 || row.cols != a.cols) throw std::runtime_error("add_row: shape mismatch");
  for (int t = 0; t < a.rows; ++t) {
    float* at = a.row(t);
    for (int d = 0; d < a.cols; ++d) at[d] += row.data[d];
  }
}

Mat layernorm(const Mat& x, const Mat& gamma, const Mat& beta, float eps) {
  if (gamma.cols != x.cols || beta.cols != x.cols) throw std::runtime_error("layernorm: shape mismatch");
  Mat y(x.rows, x.cols);
  const int D = x.cols;
  for (int t = 0; t < x.rows; ++t) {
    const float* xt = x.row(t);
    float* yt = y.row(t);
    double mean = 0.0;
    for (int d = 0; d < D; ++d) mean += xt[d];
    mean /= D;
    double var = 0.0;
    for (int d = 0; d < D; ++d) {
      const double c = xt[d] - mean;
      var += c * c;
    }
    var /= D;
    const float inv_std = static_cast<float>(1.0 / std::sqrt(var + eps));
    const float m = static_cast<float>(mean);
    for (int d = 0; d < D; ++d) yt[d] = (xt[d] - m) * inv_std * gamma.data[d] + beta.data[d];
  }
  return y;
}

void gelu_inplace(Mat& x) {
  const float k = std::sqrt(2.0f / static_cast<float>(M_PI));
  for (float& v : x.data) {
    v = 0.5f * v * (1.0f + std::tanh(k * (v + 0.044715f * v * v * v)));
  }
}

void add_inplace(Mat& a, const Mat& b) {
  if (a.rows != b.rows || a.cols != b.cols) throw std::runtime_error("add: shape mismatch");
  for (size_t i = 0; i < a.data.size(); ++i) a.data[i] += b.data[i];
}

Mat causal_self_attention(const Mat& qkv, int n_head) {
  const int T = qkv.rows;
  const int D = qkv.cols / 3;
  if (qkv.cols != 3 * D || D % n_head != 0) throw std::runtime_error("attention: shape mismatch");
  const int dh = D / n_head;  // features per head
  const float scale = 1.0f / std::sqrt(static_cast<float>(dh));

  Mat out(T, D);
  std::vector<float> prob(T);

  for (int h = 0; h < n_head; ++h) {
    const int q_off = h * dh;          // this head's slice of the query
    const int k_off = D + h * dh;      // ... of the key
    const int v_off = 2 * D + h * dh;  // ... of the value
    for (int i = 0; i < T; ++i) {
      const float* qi = qkv.row(i) + q_off;

      // 1. Scores against every token this one is allowed to see (j <= i).
      float max_score = -INFINITY;
      for (int j = 0; j <= i; ++j) {
        const float* kj = qkv.row(j) + k_off;
        float s = 0.0f;
        for (int d = 0; d < dh; ++d) s += qi[d] * kj[d];
        s *= scale;
        prob[j] = s;
        if (s > max_score) max_score = s;
      }

      // 2. Softmax (subtract the max first so exp() cannot overflow).
      float sum = 0.0f;
      for (int j = 0; j <= i; ++j) {
        prob[j] = std::exp(prob[j] - max_score);
        sum += prob[j];
      }
      const float inv_sum = 1.0f / sum;

      // 3. Weighted sum of the values.
      float* oi = out.row(i) + q_off;
      for (int j = 0; j <= i; ++j) {
        const float p = prob[j] * inv_sum;
        const float* vj = qkv.row(j) + v_off;
        for (int d = 0; d < dh; ++d) oi[d] += p * vj[d];
      }
    }
  }
  return out;
}

}  // namespace mr
