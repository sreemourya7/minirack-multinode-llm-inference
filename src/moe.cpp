#include "moe.h"

#include <cmath>
#include <stdexcept>

#include "ops.h"

namespace mr {

Routing moe_route(const Mat& x, const Mat& router_w, const Mat& router_b, const MoeConfig& cfg) {
  const int T = x.rows, E = cfg.n_expert, k = cfg.top_k;
  if (router_w.cols != E || k < 1 || k > E) throw std::runtime_error("moe_route: bad router shape or top_k");
  const Mat scores = linear(x, router_w, router_b);  // [T, E]

  Routing r;
  r.T = T;
  r.k = k;
  r.expert.resize(static_cast<size_t>(T) * k);
  r.gate.resize(static_cast<size_t>(T) * k);
  std::vector<char> taken(E);

  for (int t = 0; t < T; ++t) {
    const float* s = scores.row(t);
    int32_t* chosen = &r.expert[static_cast<size_t>(t) * k];
    float* gate = &r.gate[static_cast<size_t>(t) * k];

    // Top-k by repeated "pick the best one left". E is small, so this is fine.
    // Strict '>' means a tie keeps the lower expert id.
    std::fill(taken.begin(), taken.end(), 0);
    for (int j = 0; j < k; ++j) {
      int best = -1;
      for (int e = 0; e < E; ++e) {
        if (!taken[e] && (best < 0 || s[e] > s[best])) best = e;
      }
      taken[best] = 1;
      chosen[j] = best;
    }

    if (cfg.softmax_gate) {
      // Softmax over just the k chosen scores. chosen[0] has the highest score.
      const float top = s[chosen[0]];
      float sum = 0.0f;
      for (int j = 0; j < k; ++j) {
        gate[j] = std::exp(s[chosen[j]] - top);
        sum += gate[j];
      }
      for (int j = 0; j < k; ++j) gate[j] /= sum;
    } else {
      for (int j = 0; j < k; ++j) gate[j] = 1.0f;
    }
  }
  return r;
}

Mat moe_run_local(const Mat& x, const Routing& route, const std::vector<Expert>& experts, int expert_begin,
                  uint64_t* rows_run) {
  const int T = x.rows, D = x.cols, k = route.k;
  Mat y(T, D);
  std::vector<int> rows;      // tokens routed to the current expert
  std::vector<float> gates;   // and their gate values

  for (size_t le = 0; le < experts.size(); ++le) {
    const int32_t e = expert_begin + static_cast<int32_t>(le);

    // Which tokens chose this expert?
    rows.clear();
    gates.clear();
    for (int t = 0; t < T; ++t) {
      for (int j = 0; j < k; ++j) {
        if (route.expert[static_cast<size_t>(t) * k + j] == e) {
          rows.push_back(t);
          gates.push_back(route.gate[static_cast<size_t>(t) * k + j]);
        }
      }
    }
    if (rows.empty()) continue;  // nobody chose it: no work at all
    if (rows_run) *rows_run += rows.size();

    // Pack just those tokens into a small matrix and run the expert on it.
    Mat xs(static_cast<int>(rows.size()), D);
    for (size_t i = 0; i < rows.size(); ++i) {
      const float* src = x.row(rows[i]);
      float* dst = xs.row(static_cast<int>(i));
      for (int d = 0; d < D; ++d) dst[d] = src[d];
    }
    const Expert& ex = experts[le];
    Mat up = linear(xs, ex.fc_w, ex.fc_b);
    gelu_inplace(up);
    const Mat ys = matmul(up, ex.mproj_w);

    // Scatter the results back to their token rows, weighted by the gate.
    for (size_t i = 0; i < rows.size(); ++i) {
      const float g = gates[i];
      const float* src = ys.row(static_cast<int>(i));
      float* dst = y.row(rows[i]);
      for (int d = 0; d < D; ++d) dst[d] += g * src[d];
    }
  }
  return y;
}

}  // namespace mr
