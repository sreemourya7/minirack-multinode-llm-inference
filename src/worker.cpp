// worker.cpp - one "node" of the rack.
//
// A worker owns a shard of the model (a range of layers, maybe the embedding,
// maybe the head). It waits for a request, runs its layers, and then either
//   - hands the hidden state to the next worker and relays that worker's
//     reply back, or
//   - if it holds the head, turns the hidden state into logits and replies.
//
//   client --tokens--> worker 0 --hidden--> worker 1 --hidden--> ... last
//   client <--logits-- worker 0 <--logits-- worker 1 <--logits-- ... last
//
// Part 1 is deliberately simple: one request at a time, blocking sockets.
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <thread>

#include "model.h"
#include "net.h"
#include "weights.h"

namespace {

struct Options {
  int port = 0;
  std::string weights;
  int first_layer = 0;
  int last_layer = 0;
  bool embed = false;
  bool head = false;
  std::string next_host;
  int next_port = 0;
};

[[noreturn]] void usage(const char* msg) {
  std::fprintf(stderr,
               "error: %s\n\n"
               "usage: mr_worker --port P --weights FILE --layers A:B [--embed] [--head] [--next HOST:PORT]\n"
               "  --layers A:B   run layers A (inclusive) to B (exclusive)\n"
               "  --embed        this worker holds the embedding (it is first in the chain)\n"
               "  --head         this worker holds the output head (it is last in the chain)\n"
               "  --next H:P     where to send the hidden state (required unless --head)\n",
               msg);
  std::exit(2);
}

Options parse(int argc, char** argv) {
  Options o;
  bool have_layers = false;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto value = [&]() -> std::string {
      if (i + 1 >= argc) usage(("missing value for " + a).c_str());
      return argv[++i];
    };
    if (a == "--port") {
      o.port = std::atoi(value().c_str());
    } else if (a == "--weights") {
      o.weights = value();
    } else if (a == "--layers") {
      const std::string v = value();
      const size_t c = v.find(':');
      if (c == std::string::npos) usage("--layers must look like A:B");
      o.first_layer = std::atoi(v.substr(0, c).c_str());
      o.last_layer = std::atoi(v.substr(c + 1).c_str());
      have_layers = true;
    } else if (a == "--embed") {
      o.embed = true;
    } else if (a == "--head") {
      o.head = true;
    } else if (a == "--next") {
      const std::string v = value();
      const size_t c = v.rfind(':');
      if (c == std::string::npos) usage("--next must look like HOST:PORT");
      o.next_host = v.substr(0, c);
      o.next_port = std::atoi(v.substr(c + 1).c_str());
    } else {
      usage(("unknown argument " + a).c_str());
    }
  }
  if (o.port <= 0) usage("--port is required");
  if (o.weights.empty()) usage("--weights is required");
  if (!have_layers) usage("--layers is required");
  if (!o.head && o.next_host.empty()) usage("a worker without --head needs --next");
  if (o.head && !o.next_host.empty()) usage("a worker with --head is last and cannot have --next");
  return o;
}

double ms_since(std::chrono::steady_clock::time_point t0) {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

// Connection to the next worker. Opened on first use and kept open; if it
// breaks we drop it and reconnect on the next request.
struct NextHop {
  std::string host;
  int port = 0;
  int fd = -1;

  bool ensure_connected() {
    if (fd >= 0) return true;
    // The next worker may still be loading its weights, so retry for a while.
    for (int attempt = 0; attempt < 50 && fd < 0; ++attempt) {
      fd = mr::connect_to(host, port);
      if (fd < 0) std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    return fd >= 0;
  }
  void drop() {
    mr::close_fd(fd);
    fd = -1;
  }
};

// Handles one request and returns the reply to send back upstream.
mr::Msg handle(const mr::Shard& shard, NextHop& next, const mr::Msg& req) {
  const auto t0 = std::chrono::steady_clock::now();

  // 1. Get a hidden state, either from tokens or from the previous worker.
  mr::Mat h;
  if (req.type == mr::MSG_TOKENS) {
    if (!shard.has_embed) return mr::make_error(req.request_id, "got tokens, but this worker has no embedding");
    h = mr::embed(shard, mr::decode_tokens(req.payload));
  } else if (req.type == mr::MSG_HIDDEN) {
    if (shard.has_embed) return mr::make_error(req.request_id, "got a hidden state, but this worker expects tokens");
    h = mr::decode_mat(req.payload);
  } else {
    return mr::make_error(req.request_id, "unexpected message type");
  }
  const int T = h.rows;

  // 2. Run this shard's layers.
  mr::run_blocks(shard, h);

  // 3a. Last worker: produce logits.
  if (shard.has_head) {
    mr::Mat logits = mr::lm_head(shard, h, (req.flags & mr::FLAG_LAST_ONLY) != 0);
    mr::Msg reply;
    reply.type = mr::MSG_LOGITS;
    reply.flags = req.flags;
    reply.request_id = req.request_id;
    reply.payload = mr::encode_mat(logits);
    std::fprintf(stderr, "[worker L%d-%d] req %llu  T=%d  compute %.1f ms (incl. head)\n", shard.first_layer,
                 shard.last_layer, static_cast<unsigned long long>(req.request_id), T, ms_since(t0));
    return reply;
  }

  // 3b. Otherwise: pass the hidden state on and wait for the reply.
  const double compute_ms = ms_since(t0);
  const auto t1 = std::chrono::steady_clock::now();
  if (!next.ensure_connected()) return mr::make_error(req.request_id, "cannot reach the next worker");
  mr::Msg fwd;
  fwd.type = mr::MSG_HIDDEN;
  fwd.flags = req.flags;
  fwd.request_id = req.request_id;
  fwd.payload = mr::encode_mat(h);
  mr::Msg reply;
  if (!mr::send_msg(next.fd, fwd) || !mr::recv_msg(next.fd, reply)) {
    next.drop();
    return mr::make_error(req.request_id, "lost the connection to the next worker");
  }
  if (reply.request_id != req.request_id) {
    next.drop();
    return mr::make_error(req.request_id, "next worker answered a different request");
  }
  std::fprintf(stderr, "[worker L%d-%d] req %llu  T=%d  compute %.1f ms  downstream %.1f ms\n", shard.first_layer,
               shard.last_layer, static_cast<unsigned long long>(req.request_id), T, compute_ms, ms_since(t1));
  return reply;
}

}  // namespace

int main(int argc, char** argv) {
  const Options opt = parse(argc, argv);
  try {
    mr::WeightFile wf(opt.weights);
    const mr::Shard shard = mr::load_shard(wf, opt.first_layer, opt.last_layer, opt.embed, opt.head);
    NextHop next{opt.next_host, opt.next_port, -1};

    const int listen_fd = mr::listen_on(opt.port);
    std::fprintf(stderr, "[worker L%d-%d] ready on port %d  embed=%d head=%d  shard=%.1f MB\n", shard.first_layer,
                 shard.last_layer, opt.port, shard.has_embed, shard.has_head, wf.bytes_loaded() / 1e6);

    for (;;) {
      const int fd = mr::accept_one(listen_fd);
      if (fd < 0) continue;
      mr::Msg req;
      while (mr::recv_msg(fd, req)) {
        mr::Msg reply;
        try {
          reply = handle(shard, next, req);
        } catch (const std::exception& e) {
          // A bad request must not take the worker down: report it and carry on.
          reply = mr::make_error(req.request_id, e.what());
        }
        if (!mr::send_msg(fd, reply)) break;
      }
      mr::close_fd(fd);
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "fatal: %s\n", e.what());
    return 1;
  }
}
