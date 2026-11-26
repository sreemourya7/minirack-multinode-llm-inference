// rank.cpp - one rank of the tensor-parallel rack (Part 2).
//
// Every rank runs the same program below on its own slice of the weights:
//
//     h = input                       (root embeds the tokens and broadcasts h)
//     tp_forward(h)                   (12 layers, 2 all-reduces each)
//     check_identical(h)              (are all copies of h still the same?)
//     gather_cols(my slice of logits) (root ends up with the full logits)
//
// Rank 0 is the root: it also talks to the client.
//
//   root:   mr_rank --rank 0 --nranks N --weights FILE --port P
//   others: mr_rank --rank R --nranks N --weights FILE --root HOST:PORT
//
// Add --moe FILE (the same file on every rank) to turn each layer's MLP into
// a Mixture of Experts (Part 3).
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>

#include "mesh.h"
#include "model.h"
#include "net.h"
#include "tp_model.h"
#include "weights.h"

namespace {

struct Options {
  int rank = -1;
  int nranks = 0;
  std::string weights;
  std::string moe;
  int port = 0;
  std::string root_host;
  int root_port = 0;
};

[[noreturn]] void usage(const char* msg) {
  std::fprintf(stderr,
               "error: %s\n\n"
               "usage: mr_rank --rank R --nranks N --weights FILE [--moe FILE] (--port P | --root HOST:PORT)\n"
               "  rank 0 is the root: give it --port (clients and other ranks connect there)\n"
               "  every other rank: give it --root HOST:PORT\n",
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
      o.port = std::atoi(value().c_str());
    } else if (a == "--root") {
      const std::string v = value();
      const size_t c = v.rfind(':');
      if (c == std::string::npos) usage("--root must look like HOST:PORT");
      o.root_host = v.substr(0, c);
      o.root_port = std::atoi(v.substr(c + 1).c_str());
    } else {
      usage(("unknown argument " + a).c_str());
    }
  }
  if (o.nranks < 1) usage("--nranks is required");
  if (o.rank < 0 || o.rank >= o.nranks) usage("--rank must be in [0, nranks)");
  if (o.weights.empty()) usage("--weights is required");
  if (o.rank == 0 && o.port <= 0) usage("the root (rank 0) needs --port");
  if (o.rank != 0 && o.root_host.empty()) usage("a non-root rank needs --root");
  return o;
}

double ms_since(std::chrono::steady_clock::time_point t0) {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

void log_request(const mr::Mesh& mesh, uint64_t id, int T, double total_ms, const mr::TpStats& st) {
  const mr::CommStats& c = mesh.stats();
  std::string load;  // tokens routed to each expert, e.g. "31,12,40,..."
  for (uint64_t n : st.expert_load) load += (load.empty() ? "" : ",") + std::to_string(n);
  std::fprintf(stderr,
               "[rank %d/%d] req %llu T=%d total_ms=%.1f compute_ms=%.1f collective_ms=%.1f collectives=%llu "
               "sent_kb=%.1f recv_kb=%.1f expert_rows=%llu expert_load=[%s]\n",
               mesh.rank(), mesh.size(), static_cast<unsigned long long>(id), T, total_ms, total_ms - c.seconds * 1e3,
               c.seconds * 1e3, static_cast<unsigned long long>(c.calls), c.bytes_sent / 1e3, c.bytes_recv / 1e3,
               static_cast<unsigned long long>(st.expert_rows), load.c_str());
}

// The program every rank runs once it has the input h. On the root, `logits`
// comes back filled; elsewhere it stays empty.
void run_request(const mr::TpShard& shard, mr::Mesh& mesh, mr::Mat& h, uint32_t flags, mr::Mat& logits,
                 mr::TpStats& stats) {
  mr::tp_forward(shard, mesh, h, &stats);
  // Test hook: MR_TEST_CORRUPT_RANK=R makes rank R damage its copy of h, so
  // the test suite can prove that check_identical really catches it.
  if (const char* v = std::getenv("MR_TEST_CORRUPT_RANK"); v && std::atoi(v) == mesh.rank()) h.data[0] += 1e-3f;
  mesh.check_identical(h);
  const mr::Mat local = mr::tp_head_local(shard, h, (flags & mr::FLAG_LAST_ONLY) != 0);
  mesh.gather_cols(local, logits);
}

int serve_as_root(const Options& opt, const mr::TpShard& shard, mr::Mesh& mesh) {
  const int listen_fd = mr::listen_on(opt.port);
  mesh.accept_peers(listen_fd);
  std::fprintf(stderr, "[rank 0/%d] ready on port %d  heads %d-%d  units %d-%d  vocab %d-%d  weights=%.1f MB\n",
               mesh.size(), opt.port, shard.head_begin, shard.head_end, shard.unit_begin, shard.unit_end,
               shard.vocab_begin, shard.vocab_end, shard.resident_bytes / 1e6);

  bool mesh_ok = true;
  for (;;) {
    const int fd = mr::accept_one(listen_fd);
    if (fd < 0) continue;
    mr::Msg req;
    while (mr::recv_msg(fd, req)) {
      mr::Msg reply;
      reply.request_id = req.request_id;
      if (req.type == mr::MSG_PING) {
        reply.type = mr::MSG_PONG;
      } else if (req.type != mr::MSG_TOKENS) {
        reply = mr::make_error(req.request_id, "unexpected message type");
      } else if (!mesh_ok) {
        reply = mr::make_error(req.request_id, "the rack is down: a rank was lost (restart it; recovery is a later part)");
      } else {
        const auto t0 = std::chrono::steady_clock::now();
        mr::Mat h;
        bool started = false;
        try {
          // Validate and embed BEFORE telling the other ranks anything, so a
          // bad request never reaches them.
          h = mr::embed_tokens(shard.cfg, shard.wte, shard.wpe, mr::decode_tokens(req.payload));
          const int T = h.rows;
          mesh.reset_stats();
          started = true;
          mesh.start_request(req.request_id, req.flags, h);
          mr::Mat logits;
          mr::TpStats stats;
          run_request(shard, mesh, h, req.flags, logits, stats);
          reply.type = mr::MSG_LOGITS;
          reply.flags = req.flags;
          reply.payload = mr::encode_mat(logits);
          log_request(mesh, req.request_id, T, ms_since(t0), stats);
        } catch (const std::exception& e) {
          reply = mr::make_error(req.request_id, e.what());
          // A failure after the request started means the ranks are no longer
          // in step with each other. Stop using the mesh rather than guess.
          if (started) {
            mesh_ok = false;
            std::fprintf(stderr, "[rank 0/%d] rack failed during req %llu: %s\n", mesh.size(),
                         static_cast<unsigned long long>(req.request_id), e.what());
          }
        }
      }
      if (!mr::send_msg(fd, reply)) break;
    }
    mr::close_fd(fd);
  }
}

int serve_as_rank(const Options& opt, const mr::TpShard& shard, mr::Mesh& mesh) {
  mesh.connect_to_root(opt.root_host, opt.root_port);
  std::fprintf(stderr, "[rank %d/%d] ready  heads %d-%d  units %d-%d  vocab %d-%d  weights=%.1f MB\n", mesh.rank(),
               mesh.size(), shard.head_begin, shard.head_end, shard.unit_begin, shard.unit_end, shard.vocab_begin,
               shard.vocab_end, shard.resident_bytes / 1e6);
  uint64_t id = 0;
  uint32_t flags = 0;
  mr::Mat h;
  for (;;) {
    mesh.reset_stats();
    if (!mesh.wait_request(id, flags, h)) {
      std::fprintf(stderr, "[rank %d/%d] the root is gone; exiting\n", mesh.rank(), mesh.size());
      return 1;
    }
    const auto t0 = std::chrono::steady_clock::now();
    const int T = h.rows;
    mr::Mat unused;
    mr::TpStats stats;
    run_request(shard, mesh, h, flags, unused, stats);
    log_request(mesh, id, T, ms_since(t0), stats);
  }
}

}  // namespace

int main(int argc, char** argv) {
  const Options opt = parse(argc, argv);
  try {
    mr::WeightFile wf(opt.weights);
    std::unique_ptr<mr::WeightFile> moe;
    if (!opt.moe.empty()) moe = std::make_unique<mr::WeightFile>(opt.moe);
    const mr::TpShard shard = mr::load_tp_shard(wf, opt.rank, opt.nranks, moe.get());
    if (shard.moe.enabled) {
      std::fprintf(stderr, "[rank %d/%d] MoE: %d experts, top-%d, %s gates, my experts %d-%d\n", opt.rank, opt.nranks,
                   shard.moe.n_expert, shard.moe.top_k, shard.moe.softmax_gate ? "softmax" : "unit", shard.expert_begin,
                   shard.expert_end);
    }
    mr::Mesh mesh(opt.rank, opt.nranks);
    return opt.rank == 0 ? serve_as_root(opt, shard, mesh) : serve_as_rank(opt, shard, mesh);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "[rank %d] fatal: %s\n", opt.rank, e.what());
    return 1;
  }
}
