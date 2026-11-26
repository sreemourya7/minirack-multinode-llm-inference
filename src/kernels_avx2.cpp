// kernels_avx2.cpp - the AVX2 + FMA versions of the hot loops.
//
// This is the only file compiled with -mavx2 -mfma. Keeping those flags out
// of the rest of the program means the binary still starts on a CPU without
// AVX2: the dispatcher in kernels.cpp simply never calls into this file.
//
// Vocabulary used below:
//   __m256              a register holding 8 floats
//   _mm256_loadu_ps     read 8 floats from memory
//   _mm256_set1_ps(a)   8 copies of a
//   _mm256_fmadd_ps     a * b + c on all 8 at once, rounded once ("fused")
//   _mm256_storeu_ps    write 8 floats back
//
// Because a fused multiply-add rounds once where the scalar code rounds
// twice, results differ from the scalar kernels in the last bits. That is
// expected. What must hold is that these kernels agree with THEMSELVES for
// any batch size, and the leftover elements that do not fill a register use
// std::fma so that they round the same way as the vector lanes.
#include <immintrin.h>

#include <algorithm>
#include <cmath>
#include <cstddef>

#include "kernels.h"

namespace mr::kern {

namespace {

// Adds the 8 floats of a register together, always in the same order.
inline float hsum(__m256 v) {
  const __m128 lo = _mm256_castps256_ps128(v);
  const __m128 hi = _mm256_extractf128_ps(v, 1);
  __m128 s = _mm_add_ps(lo, hi);             // 4 partial sums
  s = _mm_add_ps(s, _mm_movehl_ps(s, s));    // 2
  s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));  // 1
  return _mm_cvtss_f32(s);
}

}  // namespace

void gemm_acc_avx2(const float* x, int T, int in, const float* w, int out, float* y) {
  const int kBlock = 16;        // tokens that share one pass over w
  const int out8 = out & ~7;    // the part of a row that fills whole registers

  for (int t0 = 0; t0 < T; t0 += kBlock) {
    const int t1 = std::min(T, t0 + kBlock);
    int i = 0;

    // Four rows of w at a time. For each token, y's 8 floats are loaded once,
    // updated by four fused multiply-adds, and stored once. The four updates
    // happen in the order i, i+1, i+2, i+3, so every element of y still
    // receives its additions in the order i = 0, 1, 2, ... whatever T is.
    for (; i + 4 <= in; i += 4) {
      const float* w0 = w + static_cast<size_t>(i) * out;
      const float* w1 = w0 + out;
      const float* w2 = w1 + out;
      const float* w3 = w2 + out;
      for (int t = t0; t < t1; ++t) {
        const float* xt = x + static_cast<size_t>(t) * in + i;
        float* yt = y + static_cast<size_t>(t) * out;
        const __m256 a0 = _mm256_set1_ps(xt[0]);
        const __m256 a1 = _mm256_set1_ps(xt[1]);
        const __m256 a2 = _mm256_set1_ps(xt[2]);
        const __m256 a3 = _mm256_set1_ps(xt[3]);
        int o = 0;
        for (; o < out8; o += 8) {
          __m256 v = _mm256_loadu_ps(yt + o);
          v = _mm256_fmadd_ps(a0, _mm256_loadu_ps(w0 + o), v);
          v = _mm256_fmadd_ps(a1, _mm256_loadu_ps(w1 + o), v);
          v = _mm256_fmadd_ps(a2, _mm256_loadu_ps(w2 + o), v);
          v = _mm256_fmadd_ps(a3, _mm256_loadu_ps(w3 + o), v);
          _mm256_storeu_ps(yt + o, v);
        }
        for (; o < out; ++o) {
          float v = yt[o];
          v = std::fma(xt[0], w0[o], v);
          v = std::fma(xt[1], w1[o], v);
          v = std::fma(xt[2], w2[o], v);
          v = std::fma(xt[3], w3[o], v);
          yt[o] = v;
        }
      }
    }

    // Rows of w left over when `in` is not a multiple of 4.
    for (; i < in; ++i) {
      const float* wi = w + static_cast<size_t>(i) * out;
      for (int t = t0; t < t1; ++t) {
        const float xi = x[static_cast<size_t>(t) * in + i];
        float* yt = y + static_cast<size_t>(t) * out;
        const __m256 a = _mm256_set1_ps(xi);
        int o = 0;
        for (; o < out8; o += 8) {
          _mm256_storeu_ps(yt + o, _mm256_fmadd_ps(a, _mm256_loadu_ps(wi + o), _mm256_loadu_ps(yt + o)));
        }
        for (; o < out; ++o) yt[o] = std::fma(xi, wi[o], yt[o]);
      }
    }
  }
}

float dot_avx2(const float* a, const float* b, int n) {
  // Four independent running sums. A single running sum would force each
  // addition to wait for the one before it; four let the CPU overlap them.
  __m256 s0 = _mm256_setzero_ps(), s1 = s0, s2 = s0, s3 = s0;
  int i = 0;
  for (; i + 32 <= n; i += 32) {
    s0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), s0);
    s1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8), s1);
    s2 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 16), _mm256_loadu_ps(b + i + 16), s2);
    s3 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 24), _mm256_loadu_ps(b + i + 24), s3);
  }
  for (; i + 8 <= n; i += 8) s0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), s0);
  float r = hsum(_mm256_add_ps(_mm256_add_ps(s0, s1), _mm256_add_ps(s2, s3)));
  for (; i < n; ++i) r = std::fma(a[i], b[i], r);
  return r;
}

void axpy_avx2(float* y, const float* x, float a, int n) {
  const __m256 av = _mm256_set1_ps(a);
  int i = 0;
  for (; i + 8 <= n; i += 8) {
    _mm256_storeu_ps(y + i, _mm256_fmadd_ps(av, _mm256_loadu_ps(x + i), _mm256_loadu_ps(y + i)));
  }
  for (; i < n; ++i) y[i] = std::fma(a, x[i], y[i]);
}

void rows_dot_avx2(const float* x, int T, int D, const float* w, int V, float* out) {
  // Row v of w on the outside: with several tokens, each row of w is fetched
  // from memory once and used for all of them.
  for (int v = 0; v < V; ++v) {
    const float* wv = w + static_cast<size_t>(v) * D;
    for (int t = 0; t < T; ++t) {
      out[static_cast<size_t>(t) * V + v] = dot_avx2(x + static_cast<size_t>(t) * D, wv, D);
    }
  }
}

double sum_avx2(const float* x, long n) {
  __m256 s0 = _mm256_setzero_ps(), s1 = s0, s2 = s0, s3 = s0;
  long i = 0;
  for (; i + 32 <= n; i += 32) {
    s0 = _mm256_add_ps(s0, _mm256_loadu_ps(x + i));
    s1 = _mm256_add_ps(s1, _mm256_loadu_ps(x + i + 8));
    s2 = _mm256_add_ps(s2, _mm256_loadu_ps(x + i + 16));
    s3 = _mm256_add_ps(s3, _mm256_loadu_ps(x + i + 24));
  }
  double r = hsum(_mm256_add_ps(_mm256_add_ps(s0, s1), _mm256_add_ps(s2, s3)));
  for (; i < n; ++i) r += x[i];
  return r;
}

}  // namespace mr::kern
