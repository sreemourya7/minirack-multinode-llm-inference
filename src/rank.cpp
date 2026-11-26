// rank.cpp - one rank of the tensor-parallel rack (Part 2).
//
// Every rank runs the same program (execute, below) on its own slice of the
// weights:
//
//     h = input                       (root embeds the tokens and broadcasts h)
//     tp_forward(h)                   (12 layers, 2 all-reduces each)
//     check_identical(h)              (are all copies of h still the same?)
//     gather_cols(my slice of logits) (root ends up with the full logits)
//
// Rank 0 is the root: it also talks to the client.
//
// Requests come in four kinds (Part 4):
//     forward   a whole sequence, nothing kept
//     prefill   a prompt; its keys and values are stored in a new session
//     decode    one more token for a session, using the stored keys and values
//     release   free a session
//
//   root:   mr_rank --rank 0 --nranks N --weights FILE --port P
//   others: mr_rank --rank R --nranks N --weights FILE --root HOST:PORT
//
// Add --moe FILE (the same file on every rank) to turn each layer's MLP into
// a Mixture of Experts (Part 3).
#include <chrono>
#include <cstdio>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "kv_cache.h"
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

const char* op_name(uint32_t op) {
  switch (op) {
    case mr::OP_FORWARD: return "forward";
    case mr::OP_PREFILL: return "prefill";
    case mr::OP_DECODE: return "decode";
    case mr::OP_RELEASE: return "release";
    default: return "?";
  }
}

// What a request produced. Only the root fills logits / token.
struct Output {
  mr::Mat logits;
  int32_t token = -1;
  float token_logit = 0.0f;
  int session_len = 0;     // tokens in the session's KV cache afterwards
  uint64_t kv_bytes = 0;   // size of THIS rank's part of that cache
  double head_ms = 0.0;    // time this rank spent on the vocabulary projection
};

void log_request(const mr::Mesh& mesh, const mr::ReqInfo& info, int T, double total_ms, const mr::TpStats& st,
                 const Output& out) {
  const mr::CommStats& c = mesh.stats();
  std::string load;  // tokens routed to each expert, e.g. "31,12,40,..."
  for (uint64_t n : st.expert_load) load += (load.empty() ? "" : ",") + std::to_string(n);
  std::fprintf(stderr,
               "[rank %d/%d] req %llu op=%s T=%d len=%d kv_kb=%.1f total_ms=%.2f compute_ms=%.2f head_ms=%.2f "
               "collective_ms=%.2f collectives=%llu sent_kb=%.1f recv_kb=%.1f expert_rows=%llu expert_load=[%s]\n",
               mesh.rank(), mesh.size(), static_cast<unsigned long long>(info.request_id), op_name(info.op), T,
               out.session_len, out.kv_bytes / 1e3, total_ms, total_ms - c.seconds * 1e3, out.head_ms, c.seconds * 1e3,
               static_cast<unsigned long long>(c.calls), c.bytes_sent / 1e3, c.bytes_recv / 1e3,
               static_cast<unsigned long long>(st.expert_rows), load.c_str());
}

// The program every rank runs once the root has started a request. All ranks
// take the same branches, so they issue the same collectives in the same order.
void execute(const mr::TpShard& shard, mr::Mesh& mesh, mr::SessionTable& sessions, const mr::ReqInfo& info, mr::Mat& h,
             Output& out, mr::TpStats& stats) {
  if (info.op == mr::OP_RELEASE) {
    if (!sessions.erase(info.session)) throw std::runtime_error("release: this rank has no such session");
    return;
  }

  // Find or create the session this request works on (none for OP_FORWARD).
  mr::Session* session = nullptr;
  if (info.op == mr::OP_PREFILL) {
    const int head_size = static_cast<int>(shard.cfg.n_embd / shard.cfg.n_head);
    session = &sessions.create(info.session, static_cast<int>(shard.cfg.n_layer),
                               (shard.head_end - shard.head_begin) * head_size);
  } else if (info.op == mr::OP_DECODE) {
    session = sessions.find(info.session);
    if (!session) throw std::runtime_error("decode: this rank has no such session");
    // The root says how long it believes the session is. Agreeing on that is
    // a cheap check that every rank's cache has seen the same history.
    if (static_cast<uint32_t>(session->len) != info.pos) {
      throw std::runtime_error("decode: this rank's session length differs from the root's");
    }
  }

  mr::tp_forward(shard, mesh, h, &stats, session);
  // Test hook: MR_TEST_CORRUPT_RANK=R makes rank R damage its copy of h, so
  // the test suite can prove that check_identical really catches it.
  if (const char* v = std::getenv("MR_TEST_CORRUPT_RANK"); v && std::atoi(v) == mesh.rank()) h.data[0] += 1e-3f;
  mesh.check_identical(h);

  const bool argmax = (info.flags & mr::FLAG_ARGMAX) != 0;
  const bool last_only = argmax || (info.flags & mr::FLAG_LAST_ONLY) != 0 || info.op != mr::OP_FORWARD;
  const auto head_t0 = std::chrono::steady_clock::now();
  const mr::Mat local = mr::tp_head_local(shard, h, last_only);
  out.head_ms = ms_since(head_t0);

  if (!argmax) {
    mesh.gather_cols(local, out.logits);  // ship my slice of the logits to the root
  } else {
    // Pick the next token on the rack. Each rank finds the best token in its
    // own slice of the vocabulary and sends just (logit, token id): 8 bytes
    // instead of thousands of logits. The root keeps the best of those.
    // (A float holds integers exactly up to 16 million, far above the vocabulary size.)
    const float* row = local.row(0);
    int best = 0;
    for (int v = 1; v < local.cols; ++v) {
      if (row[v] > row[best]) best = v;
    }
    mr::Mat mine(1, 2);
    mine.data[0] = row[best];
    mine.data[1] = static_cast<float>(shard.vocab_begin + best);
    mr::Mat all;
    mesh.gather_cols(mine, all);
    if (mesh.is_root()) {
      int winner = 0;  // ranks are in vocabulary order, so '>' keeps the lowest token id on a tie
      for (int r = 1; r < mesh.size(); ++r) {
        if (all.data[2 * r] > all.data[2 * winner]) winner = r;
      }
      out.token_logit = all.data[2 * winner];
      out.token = static_cast<int32_t>(all.data[2 * winner + 1]);
    }
  }
  if (session) {
    out.session_len = session->len;
    out.kv_bytes = session->bytes();
  }
}

// Root only: turn a client message into a request (info + embedded input).
// Everything that can be wrong with a request is checked HERE, before any
// other rank hears about it. Throws on a bad request.
void prepare(const mr::TpShard& shard, mr::SessionTable& sessions, const mr::Msg& req, mr::ReqInfo& info, mr::Mat& h) {
  info.request_id = req.request_id;
  info.flags = req.flags;
  const std::vector<uint8_t>& p = req.payload;
  auto read_session = [&]() {
    if (p.size() < sizeof(uint64_t)) throw std::runtime_error("payload too short");
    std::memcpy(&info.session, p.data(), sizeof(uint64_t));
  };
  switch (req.type) {
    case mr::MSG_TOKENS:
      info.op = mr::OP_FORWARD;
      h = mr::embed_tokens(shard.cfg, shard.wte, shard.wpe, mr::decode_tokens(p));
      break;
    case mr::MSG_PREFILL:
      read_session();
      if (sessions.find(info.session)) throw std::runtime_error("session already exists");
      if (sessions.size() >= mr::SessionTable::kMaxSessions) throw std::runtime_error("too many open sessions");
      info.op = mr::OP_PREFILL;
      h = mr::embed_tokens(shard.cfg, shard.wte, shard.wpe,
                           mr::decode_tokens(std::vector<uint8_t>(p.begin() + sizeof(uint64_t), p.end())));
      break;
    case mr::MSG_DECODE: {
      read_session();
      int32_t token;
      if (p.size() != sizeof(uint64_t) + sizeof token) throw std::runtime_error("bad decode payload");
      std::memcpy(&token, p.data() + sizeof(uint64_t), sizeof token);
      const mr::Session* s = sessions.find(info.session);
      if (!s) throw std::runtime_error("no such session");
      if (s->len >= static_cast<int>(shard.cfg.n_ctx)) throw std::runtime_error("the context window is full");
      info.op = mr::OP_DECODE;
      info.pos = static_cast<uint32_t>(s->len);
      h = mr::embed_tokens(shard.cfg, shard.wte, shard.wpe, {token}, s->len);  // position = current length
      break;
    }
    case mr::MSG_RELEASE:
      read_session();
      if (!sessions.find(info.session)) throw std::runtime_error("no such session");
      info.op = mr::OP_RELEASE;
      break;
    default:
      throw std::runtime_error("unexpected message type");
  }
}

int serve_as_root(const Options& opt, const mr::TpShard& shard, mr::Mesh& mesh) {
  const int listen_fd = mr::listen_on(opt.port);
  mesh.accept_peers(listen_fd);
  std::fprintf(stderr, "[rank 0/%d] ready on port %d  heads %d-%d  units %d-%d  vocab %d-%d  weights=%.1f MB\n",
               mesh.size(), opt.port, shard.head_begin, shard.head_end, shard.unit_begin, shard.unit_end,
               shard.vocab_begin, shard.vocab_end, shard.resident_bytes / 1e6);

  mr::SessionTable sessions;
  bool mesh_ok = true;

  // Runs one request on the whole rack. Returns the reply for the client.
  auto run = [&](const mr::Msg& req) -> mr::Msg {
    mr::Msg reply;
    reply.request_id = req.request_id;
    if (req.type == mr::MSG_PING) {
      reply.type = mr::MSG_PONG;
      return reply;
    }
    if (!mesh_ok) {
      return mr::make_error(req.request_id, "the rack is down: a rank was lost (restart it; recovery is a later part)");
    }
    const auto t0 = std::chrono::steady_clock::now();
    bool started = false;
    try {
      mr::ReqInfo info;
      mr::Mat h;
      prepare(shard, sessions, req, info, h);
      const int T = h.rows;
      mesh.reset_stats();
      started = true;
      mesh.start_request(info, h);
      Output out;
      mr::TpStats stats;
      execute(shard, mesh, sessions, info, h, out, stats);

      reply.flags = req.flags;
      if (info.op == mr::OP_RELEASE) {
        reply.type = mr::MSG_OK;
      } else if (info.flags & mr::FLAG_ARGMAX) {
        reply.type = mr::MSG_TOKEN;
        reply.payload.resize(sizeof out.token + sizeof out.token_logit);
        std::memcpy(reply.payload.data(), &out.token, sizeof out.token);
        std::memcpy(reply.payload.data() + sizeof out.token, &out.token_logit, sizeof out.token_logit);
      } else {
        reply.type = mr::MSG_LOGITS;
        reply.payload = mr::encode_mat(out.logits);
      }
      log_request(mesh, info, T, ms_since(t0), stats, out);
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
    return reply;
  };

  for (;;) {
    const int fd = mr::accept_one(listen_fd);
    if (fd < 0) continue;
    std::vector<uint64_t> owned;  // sessions this client opened and has not released
    mr::Msg req;
    while (mr::recv_msg(fd, req)) {
      const mr::Msg reply = run(req);
      if (reply.type != mr::MSG_ERROR && req.payload.size() >= sizeof(uint64_t)) {
        uint64_t id;
        std::memcpy(&id, req.payload.data(), sizeof id);
        if (req.type == mr::MSG_PREFILL) owned.push_back(id);
        if (req.type == mr::MSG_RELEASE) owned.erase(std::remove(owned.begin(), owned.end(), id), owned.end());
      }
      if (!mr::send_msg(fd, reply)) break;
    }
    mr::close_fd(fd);
    // The client is gone. Free anything it left behind, on every rank, so a
    // crashed or careless client cannot leak cache memory.
    for (uint64_t id : owned) {
      mr::Msg release;
      release.type = mr::MSG_RELEASE;
      release.payload.resize(sizeof id);
      std::memcpy(release.payload.data(), &id, sizeof id);
      run(release);
    }
  }
}

int serve_as_rank(const Options& opt, const mr::TpShard& shard, mr::Mesh& mesh) {
  mesh.connect_to_root(opt.root_host, opt.root_port);
  std::fprintf(stderr, "[rank %d/%d] ready  heads %d-%d  units %d-%d  vocab %d-%d  weights=%.1f MB\n", mesh.rank(),
               mesh.size(), shard.head_begin, shard.head_end, shard.unit_begin, shard.unit_end, shard.vocab_begin,
               shard.vocab_end, shard.resident_bytes / 1e6);
  mr::SessionTable sessions;
  mr::ReqInfo info;
  mr::Mat h;
  for (;;) {
    mesh.reset_stats();
    if (!mesh.wait_request(info, h)) {
      std::fprintf(stderr, "[rank %d/%d] the root is gone; exiting\n", mesh.rank(), mesh.size());
      return 1;
    }
    const auto t0 = std::chrono::steady_clock::now();
    const int T = h.rows;
    Output out;
    mr::TpStats stats;
    execute(shard, mesh, sessions, info, h, out, stats);
    log_request(mesh, info, T, ms_since(t0), stats, out);
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
