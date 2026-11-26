// bench.cpp - microbenchmarks for the kernels, and the memory "roofline" (Part 6).
//
//     build/mr_bench
//
// The question this answers: why does one decoded token cost several times
// what one prefill token costs, when the arithmetic per token is the same?
//
// It times the matrix kernel on a GPT-2-sized matrix (768 x 3072) three ways:
//
//   hot,  32 tokens  one matrix, used for 32 tokens per pass      (like prefill)
//   hot,  1 token    one matrix, used again and again             (the kernel's own speed)
//   cold, 1 token    many different matrices, one token through   (like decode: every
//                    each, more data than the CPU caches hold      weight is read once)
//
// and it measures how fast this machine can read memory at all. A decoded
// token has to read every weight once, so
//
//     time per token  >=  bytes of weights / memory read speed
//
// no matter how good the arithmetic is. That bound is the roofline.
//
// Finally it runs the cold case in 1, 2 and 4 processes at the same time. All
// the cores of one machine share the same memory, so the read speed each
// process gets can drop when several read at once. That decides how much a
// second rank on the same machine can help decode.
//
// Output: a table for people, then key=value lines for the test script.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include <sys/wait.h>
#include <unistd.h>

#include "kernels.h"

namespace {

using Clock = std::chrono::steady_clock;
using Gemm = void (*)(const float*, int, int, const float*, int, float*);

double seconds_since(Clock::time_point t0) { return std::chrono::duration<double>(Clock::now() - t0).count(); }

void fill(std::vector<float>& v, uint32_t seed) {
  // A cheap generator: the values do not matter, only that every page is
  // really written (so the memory exists) and nothing is denormal or zero.
  uint32_t s = seed * 2654435761u + 1u;
  for (float& x : v) {
    s = s * 1664525u + 1013904223u;
    x = static_cast<float>(static_cast<int32_t>(s >> 8) % 2001 - 1000) * 1e-3f;
  }
}

struct Result {
  double gmac_per_s;  // billions of multiply-adds per second
  double gb_per_s;    // gigabytes of weights read per second
};

// One matrix, reused. T tokens per call.
Result bench_hot(Gemm f, int T, int in, int out, const std::vector<float>& w) {
  std::vector<float> x(static_cast<size_t>(T) * in), y(static_cast<size_t>(T) * out);
  fill(x, 7);
  f(x.data(), T, in, w.data(), out, y.data());  // warm up
  double best = 1e30;
  const int calls = std::max(1, 64 / T);
  for (int rep = 0; rep < 5; ++rep) {
    const auto t0 = Clock::now();
    for (int c = 0; c < calls; ++c) f(x.data(), T, in, w.data(), out, y.data());
    best = std::min(best, seconds_since(t0) / calls);
  }
  const double macs = static_cast<double>(T) * in * out;
  return {macs / best / 1e9, macs / T * 4.0 / best / 1e9};
}

// Many matrices, one token through each, in turn. By the time a matrix comes
// round again it has long been pushed out of the caches.
Result bench_cold(Gemm f, int in, int out, const std::vector<std::vector<float>>& ws) {
  std::vector<float> x(static_cast<size_t>(in)), y(static_cast<size_t>(out));
  fill(x, 9);
  double best = 1e30;
  for (int rep = 0; rep < 3; ++rep) {
    const auto t0 = Clock::now();
    for (const auto& w : ws) f(x.data(), 1, in, w.data(), out, y.data());
    best = std::min(best, seconds_since(t0));
  }
  const double macs = static_cast<double>(ws.size()) * in * out;
  return {macs / best / 1e9, macs * 4.0 / best / 1e9};
}

// Runs `procs` processes at the same moment. Each streams one token through
// its own private set of matrices, again and again (the decode pattern).
// Returns the gigabytes of weights per second that EACH process managed, on
// average, or 0 if the processes could not be started.
double bench_concurrent(Gemm f, int procs, int in, int out, int n_mats) {
  int ready[2], go[2], result[2];
  if (pipe(ready) != 0 || pipe(go) != 0 || pipe(result) != 0) return 0.0;
  std::vector<pid_t> kids;
  for (int p = 0; p < procs; ++p) {
    const pid_t pid = fork();
    if (pid < 0) break;
    if (pid == 0) {
      close(ready[0]);
      close(go[1]);
      close(result[0]);
      std::vector<std::vector<float>> ws(static_cast<size_t>(n_mats), std::vector<float>(static_cast<size_t>(in) * out));
      for (int i = 0; i < n_mats; ++i) fill(ws[static_cast<size_t>(i)], static_cast<uint32_t>(1000 * (p + 1) + i));
      std::vector<float> x(static_cast<size_t>(in)), y(static_cast<size_t>(out));
      fill(x, 9);
      for (const auto& w : ws) f(x.data(), 1, in, w.data(), out, y.data());  // touch everything once
      char c = 1;
      if (write(ready[1], &c, 1) != 1) _exit(1);
      (void)!read(go[0], &c, 1);  // returns when the parent closes its end: everyone starts together
      const int reps = 4;
      const auto t0 = Clock::now();
      for (int r = 0; r < reps; ++r) {
        for (const auto& w : ws) f(x.data(), 1, in, w.data(), out, y.data());
      }
      const double gb_s = static_cast<double>(reps) * n_mats * in * out * 4.0 / seconds_since(t0) / 1e9;
      (void)!write(result[1], &gb_s, sizeof gb_s);
      _exit(0);
    }
    kids.push_back(pid);
  }
  close(ready[1]);
  close(go[0]);
  close(result[1]);
  char c;
  size_t up = 0;
  while (up < kids.size() && read(ready[0], &c, 1) == 1) ++up;  // wait until every child has its memory ready
  close(go[1]);                                                 // go
  double sum = 0.0;
  int got = 0;
  double v;
  while (read(result[0], &v, sizeof v) == static_cast<ssize_t>(sizeof v)) {
    sum += v;
    ++got;
  }
  for (pid_t k : kids) waitpid(k, nullptr, 0);
  close(ready[0]);
  close(result[0]);
  return got == procs ? sum / got : 0.0;
}

}  // namespace

int main() {
  const bool avx2 = mr::kern::avx2_supported();
  const int in = 768, out = 3072;  // GPT-2's MLP-up matrix: 9.4 MB
  const int n_cold = 48;           // 48 of them: 453 MB, about the size of the whole model

  std::printf("allocating %d matrices of %d x %d (%.0f MB) ...\n", n_cold, in, out, n_cold * in * out * 4.0 / 1e6);
  std::vector<std::vector<float>> ws(n_cold, std::vector<float>(static_cast<size_t>(in) * out));
  for (int i = 0; i < n_cold; ++i) fill(ws[static_cast<size_t>(i)], static_cast<uint32_t>(i));

  // --- how fast can this machine read memory at all? ---
  double read_gb_s = 0.0;
  {
    std::vector<float> big(static_cast<size_t>(128) << 20);  // 512 MB
    fill(big, 99);
    double best = 1e30, sink = 0.0;
    for (int rep = 0; rep < 4; ++rep) {
      const auto t0 = Clock::now();
      if (avx2) {
        sink += mr::kern::sum_avx2(big.data(), static_cast<long>(big.size()));
      } else {
        double s = 0.0;
        for (float v : big) s += v;
        sink += s;
      }
      best = std::min(best, seconds_since(t0));
    }
    read_gb_s = big.size() * 4.0 / best / 1e9;
    std::printf("memory read speed, one plain sequential pass over 512 MB: %.1f GB/s   [checksum %.3g]\n\n", read_gb_s, sink);
  }

  struct Variant {
    const char* name;
    Gemm f;
  };
  std::vector<Variant> variants = {{"scalar", mr::kern::gemm_acc_scalar},
                                   {"scalar_blocked", mr::kern::gemm_acc_scalar_blocked}};
  if (avx2) variants.push_back({"avx2", mr::kern::gemm_acc_avx2});

  std::printf("matrix kernel, %d x %d, billions of multiply-adds per second\n", in, out);
  std::printf("%-16s%16s%16s%16s%22s\n", "kernel", "hot, 32 tokens", "hot, 1 token", "cold, 1 token", "cold: weights read");
  std::vector<std::string> lines;
  for (const Variant& v : variants) {
    const Result hot32 = bench_hot(v.f, 32, in, out, ws[0]);
    const Result hot1 = bench_hot(v.f, 1, in, out, ws[0]);
    const Result cold1 = bench_cold(v.f, in, out, ws);
    std::printf("%-16s%16.2f%16.2f%16.2f%17.1f GB/s\n", v.name, hot32.gmac_per_s, hot1.gmac_per_s, cold1.gmac_per_s,
                cold1.gb_per_s);
    char buf[256];
    std::snprintf(buf, sizeof buf, "gemm kernel=%s hot32_gmac=%.3f hot1_gmac=%.3f cold1_gmac=%.3f cold1_gb_s=%.3f", v.name,
                  hot32.gmac_per_s, hot1.gmac_per_s, cold1.gmac_per_s, cold1.gb_per_s);
    lines.push_back(buf);
  }

  // --- the vocabulary projection: 50257 dot products of length 768 over a 154 MB table ---
  {
    const int V = 50257, D = 768;
    std::vector<float> wte(static_cast<size_t>(V) * D), x(static_cast<size_t>(D)), outv(static_cast<size_t>(V));
    fill(wte, 5);
    fill(x, 6);
    std::printf("\nvocabulary projection, one token (%d x %d, %.0f MB)\n", V, D, V * D * 4.0 / 1e6);
    std::printf("%-16s%12s%22s\n", "kernel", "ms", "weights read");
    auto run = [&](const char* name, void (*f)(const float*, int, int, const float*, int, float*)) {
      double best = 1e30;
      for (int rep = 0; rep < 3; ++rep) {
        // Push the table out of the caches between runs by walking other memory.
        for (int i = 0; i < 12; ++i) outv[0] += ws[static_cast<size_t>(i * 4)][static_cast<size_t>(rep) * 4096];
        const auto t0 = Clock::now();
        f(x.data(), 1, D, wte.data(), V, outv.data());
        best = std::min(best, seconds_since(t0));
      }
      const double gb_s = V * D * 4.0 / best / 1e9;
      std::printf("%-16s%12.1f%17.1f GB/s\n", name, best * 1e3, gb_s);
      char buf[160];
      std::snprintf(buf, sizeof buf, "head kernel=%s ms=%.3f gb_s=%.3f", name, best * 1e3, gb_s);
      lines.push_back(buf);
    };
    run("scalar", mr::kern::rows_dot_scalar);
    if (avx2) run("avx2", mr::kern::rows_dot_avx2);
  }

  // --- several processes reading weights at once ---
  {
    // Give the memory back first: each child process allocates its own matrices.
    ws.clear();
    ws.shrink_to_fit();
    const Gemm best = avx2 ? mr::kern::gemm_acc_avx2 : mr::kern::gemm_acc_scalar;
    const int cores = static_cast<int>(std::thread::hardware_concurrency());
    std::printf("\nreading weights in several processes at once (cold, 1 token, %s kernel; %d cores visible)\n",
                avx2 ? "avx2" : "scalar", cores);
    std::printf("%-12s%22s%16s\n", "processes", "each process reads", "all together");
    for (int procs : {1, 2, 4}) {
      if (procs > 1 && procs > cores) continue;
      const double each = bench_concurrent(best, procs, in, out, 32);
      if (each <= 0.0) continue;
      std::printf("%-12d%17.1f GB/s%11.1f GB/s\n", procs, each, each * procs);
      char buf[160];
      std::snprintf(buf, sizeof buf, "concurrent procs=%d each_gb_s=%.3f total_gb_s=%.3f", procs, each, each * procs);
      lines.push_back(buf);
    }
  }

  std::printf("\n");
  std::printf("memory read_gb_s=%.3f\n", read_gb_s);
  for (const std::string& l : lines) std::printf("%s\n", l.c_str());
  return 0;
}
