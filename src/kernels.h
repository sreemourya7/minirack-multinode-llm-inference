// kernels.h - the few inner loops where almost all the time goes (Part 6).
//
// Everything the model computes comes down to three loops:
//
//   gemm_acc   y[t][o] += sum over i of x[t][i] * w[i][o]     every linear layer
//   rows_dot   out[t][v] = dot(x[t], w[v])                    the vocabulary projection
//   dot, axpy  short vectors                                  inside attention
//
// Each has two implementations:
//
//   scalar  the plain loops of Parts 1 to 5, unchanged. The baseline.
//   avx2    hand-written with AVX2 and FMA: 8 floats per instruction.
//
// The implementation is chosen once, at start-up, from what the CPU supports
// (or from MR_KERNELS=scalar|avx2). After that the rest of the code calls
// kern::gemm_acc and friends and does not care which one it got.
//
// One rule every implementation must keep: the result for one token must not
// depend on which other tokens are in the batch. That is what makes decoding
// one token at a time bit-identical to prefilling them all at once (Part 4),
// which recovery by replay relies on (Part 5).
#pragma once

#include <cstdint>

namespace mr::kern {

enum class Impl : uint32_t { scalar = 0, avx2 = 1 };

bool avx2_supported();   // does this CPU (and this build) have AVX2 and FMA?
void select(Impl impl);  // throws if that implementation is not available
void select_from_env();  // MR_KERNELS=scalar|avx2; otherwise the best available
Impl active();
const char* name(Impl impl);

// --- the dispatched kernels -----------------------------------------------------
// x: [T, in]   w: [in, out]   y: [T, out], all row-major. Adds into y.
void gemm_acc(const float* x, int T, int in, const float* w, int out, float* y);
// x: [T, D]   w: [V, D]   out: [T, V]. Overwrites out.
void rows_dot(const float* x, int T, int D, const float* w, int V, float* out);
float dot(const float* a, const float* b, int n);
void axpy(float* y, const float* x, float a, int n);  // y += a * x

// --- the individual implementations, for the benchmark and the tests --------------
void gemm_acc_scalar(const float* x, int T, int in, const float* w, int out, float* y);
// Same arithmetic as scalar, but loops over w's rows on the outside so that
// each row is reused for a block of tokens while it is still in the cache.
void gemm_acc_scalar_blocked(const float* x, int T, int in, const float* w, int out, float* y);
void rows_dot_scalar(const float* x, int T, int D, const float* w, int V, float* out);
float dot_scalar(const float* a, const float* b, int n);
void axpy_scalar(float* y, const float* x, float a, int n);

// Defined in kernels_avx2.cpp, which is only compiled on x86-64. Call them
// only if avx2_supported() is true.
void gemm_acc_avx2(const float* x, int T, int in, const float* w, int out, float* y);
void rows_dot_avx2(const float* x, int T, int D, const float* w, int V, float* out);
float dot_avx2(const float* a, const float* b, int n);
void axpy_avx2(float* y, const float* x, float a, int n);
double sum_avx2(const float* x, long n);  // reads a buffer as fast as possible (memory-bandwidth test)

}  // namespace mr::kern
