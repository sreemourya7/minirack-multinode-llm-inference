// kernel_test.cpp - unit tests for the kernels (Part 6).
//
// Two kinds of claim are tested:
//
//   1. Agreement. The AVX2 kernels give the same answers as the scalar ones,
//      up to the last few bits (a fused multiply-add rounds differently).
//
//   2. Batch invariance. For EVERY implementation, computing T tokens in one
//      call gives exactly the same bits as computing them one at a time. The
//      rest of the system depends on this: it is why decode equals prefill
//      (Part 4) and why recovery by replay is exact (Part 5).
//
// Shapes are deliberately awkward (sizes that are not multiples of 8 or 4,
// single rows, empty inputs) because the leftover-element code is where
// vector kernels usually go wrong.
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "kernels.h"

namespace {

using mr::kern::Impl;
using Gemm = void (*)(const float*, int, int, const float*, int, float*);
using RowsDot = void (*)(const float*, int, int, const float*, int, float*);

int g_failures = 0;

void expect(bool ok, const char* what, int a = -1, int b = -1, int c = -1) {
  if (ok) return;
  ++g_failures;
  std::printf("  FAIL: %s (%d, %d, %d)\n", what, a, b, c);
}

std::vector<float> random_vec(size_t n, std::mt19937& rng) {
  std::normal_distribution<float> d(0.0f, 1.0f);
  std::vector<float> v(n);
  for (float& x : v) x = d(rng);
  return v;
}

// Largest |a - b| relative to the size of the numbers involved.
double rel_diff(const std::vector<float>& a, const std::vector<float>& b) {
  double worst = 0.0, scale = 1.0;
  for (size_t i = 0; i < a.size(); ++i) {
    worst = std::max(worst, static_cast<double>(std::fabs(a[i] - b[i])));
    scale = std::max(scale, static_cast<double>(std::fabs(a[i])));
  }
  return worst / scale;
}

bool same_bits(const std::vector<float>& a, const std::vector<float>& b) {
  return a.size() == b.size() && (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0);
}

// T tokens in one call vs one call per token.
void check_gemm_batch_invariance(Gemm f, const char* name, int T, int in, int out, std::mt19937& rng) {
  const auto x = random_vec(static_cast<size_t>(T) * in, rng);
  const auto w = random_vec(static_cast<size_t>(in) * out, rng);
  const auto y0 = random_vec(static_cast<size_t>(T) * out, rng);  // gemm_acc ADDS, so start from something
  auto together = y0, one_by_one = y0;
  f(x.data(), T, in, w.data(), out, together.data());
  for (int t = 0; t < T; ++t) {
    f(x.data() + static_cast<size_t>(t) * in, 1, in, w.data(), out, one_by_one.data() + static_cast<size_t>(t) * out);
  }
  expect(same_bits(together, one_by_one), name, T, in, out);
}

void check_rows_dot_batch_invariance(RowsDot f, const char* name, int T, int D, int V, std::mt19937& rng) {
  const auto x = random_vec(static_cast<size_t>(T) * D, rng);
  const auto w = random_vec(static_cast<size_t>(V) * D, rng);
  std::vector<float> together(static_cast<size_t>(T) * V), one_by_one(together.size());
  f(x.data(), T, D, w.data(), V, together.data());
  for (int t = 0; t < T; ++t) {
    f(x.data() + static_cast<size_t>(t) * D, 1, D, w.data(), V, one_by_one.data() + static_cast<size_t>(t) * V);
  }
  expect(same_bits(together, one_by_one), name, T, D, V);
}

}  // namespace

int main() {
  std::mt19937 rng(1234);
  const bool avx2 = mr::kern::avx2_supported();
  std::printf("AVX2 + FMA available: %s\n", avx2 ? "yes" : "no (only the scalar kernels are tested)");

  const int shapes[][3] = {  // T, in, out
      {1, 1, 1},  {1, 5, 7},   {3, 8, 16},  {2, 4, 8},    {17, 13, 33}, {5, 7, 9},     {16, 64, 64},
      {33, 6, 31}, {1, 768, 96}, {4, 3, 100}, {32, 96, 384}, {40, 101, 67}, {1, 4, 2304}, {9, 10, 0},
      {0, 8, 8},
  };

  std::printf("gemm_acc: %zu shapes\n", sizeof shapes / sizeof shapes[0]);
  for (const auto& s : shapes) {
    const int T = s[0], in = s[1], out = s[2];
    check_gemm_batch_invariance(mr::kern::gemm_acc_scalar, "scalar gemm is not batch-invariant", T, in, out, rng);
    check_gemm_batch_invariance(mr::kern::gemm_acc_scalar_blocked, "blocked gemm is not batch-invariant", T, in, out, rng);

    const auto x = random_vec(static_cast<size_t>(T) * in, rng);
    const auto w = random_vec(static_cast<size_t>(in) * out, rng);
    const auto y0 = random_vec(static_cast<size_t>(T) * out, rng);
    auto ref = y0, blocked = y0;
    mr::kern::gemm_acc_scalar(x.data(), T, in, w.data(), out, ref.data());
    mr::kern::gemm_acc_scalar_blocked(x.data(), T, in, w.data(), out, blocked.data());
    expect(same_bits(ref, blocked), "blocked gemm differs from scalar (must be bit-identical)", T, in, out);

    if (avx2) {
      check_gemm_batch_invariance(mr::kern::gemm_acc_avx2, "avx2 gemm is not batch-invariant", T, in, out, rng);
      auto fast = y0;
      mr::kern::gemm_acc_avx2(x.data(), T, in, w.data(), out, fast.data());
      expect(rel_diff(ref, fast) < 1e-4, "avx2 gemm disagrees with scalar", T, in, out);
    }
  }

  std::printf("dot, axpy: lengths 0 to 100\n");
  for (int n = 0; n <= 100; ++n) {
    const auto a = random_vec(static_cast<size_t>(n), rng);
    const auto b = random_vec(static_cast<size_t>(n), rng);
    if (!avx2) continue;
    const float ds = mr::kern::dot_scalar(a.data(), b.data(), n);
    const float dv = mr::kern::dot_avx2(a.data(), b.data(), n);
    expect(std::fabs(ds - dv) <= 1e-4f * (1.0f + std::fabs(ds)) * std::sqrt(static_cast<float>(n + 1)), "avx2 dot disagrees with scalar", n);
    auto ys = b, yv = b;
    mr::kern::axpy_scalar(ys.data(), a.data(), 0.37f, n);
    mr::kern::axpy_avx2(yv.data(), a.data(), 0.37f, n);
    expect(rel_diff(ys, yv) < 1e-5, "avx2 axpy disagrees with scalar", n);
  }

  const int dot_shapes[][3] = {{1, 768, 50}, {7, 64, 33}, {3, 37, 11}, {32, 100, 5}, {1, 1, 1}, {2, 31, 0}};  // T, D, V
  std::printf("rows_dot: %zu shapes\n", sizeof dot_shapes / sizeof dot_shapes[0]);
  for (const auto& s : dot_shapes) {
    const int T = s[0], D = s[1], V = s[2];
    check_rows_dot_batch_invariance(mr::kern::rows_dot_scalar, "scalar rows_dot is not batch-invariant", T, D, V, rng);
    if (!avx2) continue;
    check_rows_dot_batch_invariance(mr::kern::rows_dot_avx2, "avx2 rows_dot is not batch-invariant", T, D, V, rng);
    const auto x = random_vec(static_cast<size_t>(T) * D, rng);
    const auto w = random_vec(static_cast<size_t>(V) * D, rng);
    std::vector<float> ref(static_cast<size_t>(T) * V), fast(ref.size());
    mr::kern::rows_dot_scalar(x.data(), T, D, w.data(), V, ref.data());
    mr::kern::rows_dot_avx2(x.data(), T, D, w.data(), V, fast.data());
    expect(rel_diff(ref, fast) < 1e-4, "avx2 rows_dot disagrees with scalar", T, D, V);
  }

  // The dispatcher must refuse what it cannot run, and switch cleanly.
  mr::kern::select(Impl::scalar);
  expect(mr::kern::active() == Impl::scalar, "select(scalar) did not take effect");
  if (avx2) {
    mr::kern::select(Impl::avx2);
    expect(mr::kern::active() == Impl::avx2, "select(avx2) did not take effect");
  }

  if (g_failures) {
    std::printf("FAILED: %d kernel check(s)\n", g_failures);
    return 1;
  }
  std::printf("PASSED: kernels agree with the scalar reference and are batch-invariant\n");
  return 0;
}
