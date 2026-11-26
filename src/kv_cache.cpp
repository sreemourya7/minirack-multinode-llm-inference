#include "kv_cache.h"

#include <cmath>
#include <stdexcept>

namespace mr {

Session& SessionTable::create(uint64_t id, int n_layer, int width) {
  if (sessions_.count(id)) throw std::runtime_error("session already exists");
  if (sessions_.size() >= kMaxSessions) throw std::runtime_error("too many open sessions");
  Session& s = sessions_[id];
  s.width = width;
  s.layers.resize(static_cast<size_t>(n_layer));
  return s;
}

Session* SessionTable::find(uint64_t id) {
  auto it = sessions_.find(id);
  return it == sessions_.end() ? nullptr : &it->second;
}

bool SessionTable::erase(uint64_t id) { return sessions_.erase(id) > 0; }

Mat attend_cached(const Mat& qkv, int n_head, KvLayer& cache, int old_len) {
  const int Tn = qkv.rows;
  const int L = qkv.cols / 3;
  if (qkv.cols != 3 * L || L % n_head != 0) throw std::runtime_error("attend_cached: shape mismatch");
  if (cache.k.size() != static_cast<size_t>(old_len) * L) throw std::runtime_error("attend_cached: cache size is wrong");
  const int dh = L / n_head;
  const float scale = 1.0f / std::sqrt(static_cast<float>(dh));

  // 1. Append the new tokens' keys and values.
  for (int i = 0; i < Tn; ++i) {
    const float* row = qkv.row(i);
    cache.k.insert(cache.k.end(), row + L, row + 2 * L);
    cache.v.insert(cache.v.end(), row + 2 * L, row + 3 * L);
  }
  const float* K = cache.k.data();
  const float* V = cache.v.data();

  // 2. Each new token attends over stored tokens 0..its own position.
  Mat out(Tn, L);
  std::vector<float> prob(static_cast<size_t>(old_len) + Tn);
  for (int h = 0; h < n_head; ++h) {
    const int off = h * dh;
    for (int i = 0; i < Tn; ++i) {
      const int pos = old_len + i;  // this token's position in the sequence
      const float* qi = qkv.row(i) + off;

      float max_score = -INFINITY;
      for (int j = 0; j <= pos; ++j) {
        const float* kj = K + static_cast<size_t>(j) * L + off;
        float s = 0.0f;
        for (int d = 0; d < dh; ++d) s += qi[d] * kj[d];
        s *= scale;
        prob[j] = s;
        if (s > max_score) max_score = s;
      }

      float sum = 0.0f;
      for (int j = 0; j <= pos; ++j) {
        prob[j] = std::exp(prob[j] - max_score);
        sum += prob[j];
      }
      const float inv_sum = 1.0f / sum;

      float* oi = out.row(i) + off;
      for (int j = 0; j <= pos; ++j) {
        const float p = prob[j] * inv_sum;
        const float* vj = V + static_cast<size_t>(j) * L + off;
        for (int d = 0; d < dh; ++d) oi[d] += p * vj[d];
      }
    }
  }
  return out;
}

}  // namespace mr
