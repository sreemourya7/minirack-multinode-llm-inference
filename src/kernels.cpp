#include "kernels.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

namespace mr::kern {

// ---------------------------------------------------------------- scalar kernels
// These are the loops of Parts 1 to 5, moved here unchanged.

void gemm_acc_scalar(const float* x, int T, int in, const float* w, int out, float* y) {
  for (int t = 0; t < T; ++t) {
    float* yt = y + static_cast<size_t>(t) * out;
    const float* xt = x + static_cast<size_t>(t) * in;
    // Loop order t, i, o: the inner loop walks one row of w and one row of y,
    // both contiguous in memory.
    for (int i = 0; i < in; ++i) {
      const float xi = xt[i];
      const float* wi = w + static_cast<size_t>(i) * out;
      for (int o = 0; o < out; ++o) yt[o] += xi * wi[o];
    }
  }
}

void gemm_acc_scalar_blocked(const float* x, int T, int in, const float* w, int out, float* y) {
  // Order: block of tokens, i, t, o. For every element y[t][o] the additions
  // still happen for i = 0, 1, 2, ... in turn, so the result is bit-identical
  // to gemm_acc_scalar. What changes is memory traffic: row i of w is used
  // for a whole block of tokens before the next row is fetched.
  const int kBlock = 16;
  for (int t0 = 0; t0 < T; t0 += kBlock) {
    const int t1 = std::min(T, t0 + kBlock);
    for (int i = 0; i < in; ++i) {
      const float* wi = w + static_cast<size_t>(i) * out;
      for (int t = t0; t < t1; ++t) {
        const float xi = x[static_cast<size_t>(t) * in + i];
        float* yt = y + static_cast<size_t>(t) * out;
        for (int o = 0; o < out; ++o) yt[o] += xi * wi[o];
      }
    }
  }
}

void rows_dot_scalar(const float* x, int T, int D, const float* w, int V, float* out) {
  for (int t = 0; t < T; ++t) {
    const float* xt = x + static_cast<size_t>(t) * D;
    float* ot = out + static_cast<size_t>(t) * V;
    for (int v = 0; v < V; ++v) {
      const float* wv = w + static_cast<size_t>(v) * D;
      float acc = 0.0f;
      for (int d = 0; d < D; ++d) acc += xt[d] * wv[d];
      ot[v] = acc;
    }
  }
}

float dot_scalar(const float* a, const float* b, int n) {
  float s = 0.0f;
  for (int i = 0; i < n; ++i) s += a[i] * b[i];
  return s;
}

void axpy_scalar(float* y, const float* x, float a, int n) {
  for (int i = 0; i < n; ++i) y[i] += a * x[i];
}

// ------------------------------------------------------------------- dispatch

namespace {

using GemmFn = void (*)(const float*, int, int, const float*, int, float*);
using RowsDotFn = void (*)(const float*, int, int, const float*, int, float*);
using DotFn = float (*)(const float*, const float*, int);
using AxpyFn = void (*)(float*, const float*, float, int);

Impl g_impl = Impl::scalar;
GemmFn g_gemm = gemm_acc_scalar;
RowsDotFn g_rows_dot = rows_dot_scalar;
DotFn g_dot = dot_scalar;
AxpyFn g_axpy = axpy_scalar;

}  // namespace

bool avx2_supported() {
#if defined(MR_HAVE_AVX2)
  return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
#else
  return false;
#endif
}

void select(Impl impl) {
  if (impl == Impl::avx2) {
#if defined(MR_HAVE_AVX2)
    if (!avx2_supported()) throw std::runtime_error("this CPU does not support AVX2 and FMA");
    g_gemm = gemm_acc_avx2;
    g_rows_dot = rows_dot_avx2;
    g_dot = dot_avx2;
    g_axpy = axpy_avx2;
#else
    throw std::runtime_error("this build has no AVX2 kernels");
#endif
  } else {
    g_gemm = gemm_acc_scalar;
    g_rows_dot = rows_dot_scalar;
    g_dot = dot_scalar;
    g_axpy = axpy_scalar;
  }
  g_impl = impl;
}

void select_from_env() {
  const char* v = std::getenv("MR_KERNELS");
  if (v && std::strcmp(v, "scalar") == 0) {
    select(Impl::scalar);
  } else if (v && std::strcmp(v, "avx2") == 0) {
    select(Impl::avx2);
  } else if (v && *v) {
    throw std::runtime_error(std::string("MR_KERNELS must be scalar or avx2, not ") + v);
  } else {
    select(avx2_supported() ? Impl::avx2 : Impl::scalar);
  }
}

Impl active() { return g_impl; }
const char* name(Impl impl) { return impl == Impl::avx2 ? "avx2" : "scalar"; }

void gemm_acc(const float* x, int T, int in, const float* w, int out, float* y) { g_gemm(x, T, in, w, out, y); }
void rows_dot(const float* x, int T, int D, const float* w, int V, float* out) { g_rows_dot(x, T, D, w, V, out); }
float dot(const float* a, const float* b, int n) { return g_dot(a, b, n); }
void axpy(float* y, const float* x, float a, int n) { g_axpy(y, x, a, n); }

}  // namespace mr::kern
