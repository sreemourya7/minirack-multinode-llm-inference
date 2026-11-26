// rank.cpp - one rank of the tensor-parallel rack (Parts 2 to 5).
//
// Every rank runs the same program (execute, in exec.cpp) on its own slice of
// the weights:
//
//     h = input                       (root embeds the tokens and broadcasts h)
//     tp_forward(h)                   (12 layers, 2 all-reduces each)
//     check_identical(h)              (are all copies of h still the same?)
//     gather_cols(my slice of logits) (root ends up with the full logits)
//
// Requests come in four kinds (Part 4):
//     forward   a whole sequence, nothing kept
//     prefill   a prompt; its keys and values are stored in a new session
//     decode    one more token for a session, using the stored keys and values
//     release   free a session
//
// Rank 0 is the root (root.cpp): it also talks to the client and repairs the
// rack when a rank is lost. This file holds main() and the loop of every
// other rank.
//
//   root:   mr_rank --rank 0 --nranks N --weights FILE --port P
//   others: mr_rank --rank R --nranks N --weights FILE --root HOST:PORT
//
// Options for every rank (use the same on all):
//   --moe FILE            turn each layer's MLP into a Mixture of Experts (Part 3)
//   MR_KERNELS=scalar|avx2 (environment)  which inner loops to use (Part 6; default: AVX2 if the CPU has it)
// Options for the root (Part 5):
//   --recovery replay|checkpoint   how a replacement rank gets its state back (default replay)
//   --rank-timeout-ms N            a rank silent for this long is treated as dead (default 10000)
//   --rejoin-timeout-ms N          how long a request waits for a replacement (default 30000)
//
// A replacement for a lost rank is simply the same command run again: it
// loads its slice of the weights, connects to the root, and is given its
// state back.
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>

#include "exec.h"
#include "kernels.h"
#include "kv_cache.h"
#include "mesh.h"
#include "root.h"
#include "tp_model.h"
#include "weights.h"

namespace {

struct Options {
  int rank = -1;
  int nranks = 0;
  std::string weights;
  std::string moe;
  std::string root_host;
  int root_port = 0;
  mr::RootOptions root;
};

[[noreturn]] void usage(const char* msg) {
  std::fprintf(stderr,
               "error: %s\n\n"
               "usage: mr_rank --rank R --nranks N --weights FILE [--moe FILE] (--port P | --root HOST:PORT)\n"
               "  rank 0 is the root: give it --port (clients and other ranks connect there)\n"
               "  every other rank: give it --root HOST:PORT\n"
               "root only: [--recovery replay|checkpoint] [--rank-timeout-ms N] [--rejoin-timeout-ms N]\n",
               msg);
  std::exit(2);
}

Options parse(int argc, char** argv) {
  Options o;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto value = [&]() -> std::string {
      if (i + 1 >= argc) usage(("missing value for " + a).c_str());
      return argv[++i];
    };
    if (a == "--rank") {
      o.rank = std::atoi(value().c_str());
    } else if (a == "--nranks") {
      o.nranks = std::atoi(value().c_str());
    } else if (a == "--weights") {
      o.weights = value();
    } else if (a == "--moe") {
      o.moe = value();
    } else if (a == "--port") {
      o.root.port = std::atoi(value().c_str());
    } else if (a == "--root") {
      const std::string v = value();
      const size_t c = v.rfind(':');
      if (c == std::string::npos) usage("--root must look like HOST:PORT");
      o.root_host = v.substr(0, c);
      o.root_port = std::atoi(v.substr(c + 1).c_str());
    } else if (a == "--recovery") {
      const std::string v = value();
      if (v != "replay" && v != "checkpoint") usage("--recovery must be replay or checkpoint");
      o.root.checkpoint = v == "checkpoint";
    } else if (a == "--rank-timeout-ms") {
      o.root.rank_timeout_ms = std::atoi(value().c_str());
    } else if (a == "--rejoin-timeout-ms") {
      o.root.rejoin_timeout_ms = std::atoi(value().c_str());
    } else {
      usage(("unknown argument " + a).c_str());
    }
  }
  if (o.nranks < 1) usage("--nranks is required");
  if (o.rank < 0 || o.rank >= o.nranks) usage("--rank must be in [0, nranks)");
  if (o.weights.empty()) usage("--weights is required");
  if (o.rank == 0 && o.root.port <= 0) usage("the root (rank 0) needs --port");
  if (o.rank != 0 && o.root_host.empty()) usage("a non-root rank needs --root");
  return o;
}

// The loop of every rank except the root: wait to be told what to do, do it.
int serve_as_rank(const Options& opt, const mr::TpShard& shard, mr::Mesh& mesh) {
  mesh.connect_to_root(opt.root_host, opt.root_port);
  std::fprintf(stderr, "[rank %d/%d] ready  heads %d-%d  units %d-%d  vocab %d-%d  weights=%.1f MB  kernels=%s\n",
               mesh.rank(), mesh.size(), shard.head_begin, shard.head_end, shard.unit_begin, shard.unit_end,
               shard.vocab_begin, shard.vocab_end, shard.resident_bytes / 1e6, mr::kern::name(mr::kern::active()));
  mr::SessionTable sessions;
  mr::ReqInfo info;
  mr::Mat h;
  for (;;) {
    mesh.reset_stats();
    if (!mesh.wait_request(info, h)) {
      // Either the root died, or it decided we were dead and closed our
      // connection. In both cases our state is no longer wanted: exit, and
      // let whoever supervises this process start a fresh one.
      std::fprintf(stderr, "[rank %d/%d] the root is gone; exiting\n", mesh.rank(), mesh.size());
      return 1;
    }
    const auto t0 = std::chrono::steady_clock::now();
    const int T = h.rows;
    mr::Output out;
    mr::TpStats stats;
    try {
      mr::execute(shard, mesh, sessions, info, h, out, stats);
    } catch (const mr::RequestAborted&) {
      // Another rank failed and the root cancelled this request. Whatever we
      // half-did will be wiped by the reset the root sends next.
      std::fprintf(stderr, "[rank %d/%d] req %llu aborted by the root\n", mesh.rank(), mesh.size(),
                   static_cast<unsigned long long>(info.request_id));
      continue;
    }
    mr::log_request(mesh, info, T, mr::ms_since(t0), stats, out);
  }
}

}  // namespace

int main(int argc, char** argv) {
  const Options opt = parse(argc, argv);
  try {
    mr::kern::select_from_env();  // scalar or AVX2 kernels (Part 6); MR_KERNELS overrides
    const auto t0 = std::chrono::steady_clock::now();
    mr::WeightFile wf(opt.weights);
    std::unique_ptr<mr::WeightFile> moe;
    if (!opt.moe.empty()) moe = std::make_unique<mr::WeightFile>(opt.moe);
    const mr::TpShard shard = mr::load_tp_shard(wf, opt.rank, opt.nranks, moe.get());
    std::fprintf(stderr, "[rank %d/%d] loaded weights load_ms=%.1f\n", opt.rank, opt.nranks, mr::ms_since(t0));
    if (shard.moe.enabled) {
      std::fprintf(stderr, "[rank %d/%d] MoE: %d experts, top-%d, %s gates, my experts %d-%d\n", opt.rank, opt.nranks,
                   shard.moe.n_expert, shard.moe.top_k, shard.moe.softmax_gate ? "softmax" : "unit", shard.expert_begin,
                   shard.expert_end);
    }
    mr::Mesh mesh(opt.rank, opt.nranks);
    return opt.rank == 0 ? mr::serve_as_root(opt.root, shard, mesh) : serve_as_rank(opt, shard, mesh);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "[rank %d] fatal: %s\n", opt.rank, e.what());
    return 1;
  }
}
