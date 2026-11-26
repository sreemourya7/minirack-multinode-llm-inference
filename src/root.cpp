// root.cpp - rank 0.
//
// The root does three jobs:
//   1. It is a rank like any other: it holds a slice of the model and runs
//      execute() for every request.
//   2. It is the front door: clients connect to it, and it checks every
//      request before the other ranks hear about it.
//   3. (Part 5) It keeps the rack healthy. It notices when a rank dies or
//      hangs, cancels the request in progress on the survivors, takes in a
//      replacement, rebuilds the lost state, and then runs the interrupted
//      request again. The client sees a pause, not an error.
//
// A replacement is just another mr_rank process started with the same rank
// number. If it arrives while the rank is missing, it is put to work at once.
// If it arrives while the rank is healthy, the root keeps it waiting as a
// STANDBY: its weights are already loaded, so when the rank later fails the
// standby takes over with no loading delay.
//
// What makes step 3 possible is one rule: a request only "happened" once it
// has finished on every rank. Until then nothing is recorded (history_,
// backups_), so after a failure the root can always return the whole rack to
// the last point where every rank agreed.
#include "root.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <deque>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "exec.h"
#include "kv_cache.h"
#include "model.h"
#include "net.h"

namespace mr {

namespace {

using Clock = std::chrono::steady_clock;
constexpr int kStandby = 1000;  // event-loop tag: kStandby + r is rank r's standby socket

// A request that was wrong from the start. Nothing was sent to any rank.
struct BadRequest : std::runtime_error {
  using std::runtime_error::runtime_error;
};

class Root {
 public:
  Root(const RootOptions& opt, const TpShard& shard, Mesh& mesh) : opt_(opt), shard_(shard), mesh_(mesh) {}
  int serve();

 private:
  struct Pending {  // a client that connected while we were busy, and the first thing it said
    int fd;
    Msg first;
  };
  enum class Heal { ok, still_missing, failed_again };

  // connections
  void identify(int fd);
  bool pump_newcomers(int timeout_ms);
  void serve_client_message(const Msg& req);
  void drop_client();

  // requests
  Msg run(const Msg& req);
  Msg run_once(const Msg& req);
  void prepare(const Msg& req, ReqInfo& info, Mat& h, std::vector<int32_t>& tokens);
  void start_internal(ReqInfo& info, const Mat& h);

  // failures
  void on_rank_lost(const RankLost& e, const char* during, double blocked_ms);
  Heal heal(int wait_ms);

  const RootOptions& opt_;
  const TpShard& shard_;
  Mesh& mesh_;
  int listen_fd_ = -1;

  SessionTable sessions_;                               // my own part of every session's cache
  std::map<uint64_t, std::vector<int32_t>> history_;    // the committed tokens of every open session
  std::map<uint64_t, std::vector<Mat>> backups_;        // checkpoint mode: backups_[session][rank] = its cache rows

  bool needs_heal_ = false;  // a rank is missing, or was replaced and has not been given its state yet
  bool broken_ = false;      // a failure we do not know how to recover from
  Clock::time_point failed_at_;
  uint64_t next_internal_id_ = 1ull << 62;  // request ids for the root's own housekeeping requests

  std::vector<int> newcomers_;   // connected, but have not yet said who they are
  std::map<int, int> standby_;   // rank -> socket of a spare process waiting to replace it
  std::deque<Pending> pending_;
  int client_ = -1;              // the client being served; one at a time
  std::vector<uint64_t> owned_;  // sessions that client opened and has not released
};

// ============================================================== connections

// A new connection has sent its first message; work out who it is from that.
// A replacement rank says HELLO. Anything else is a client.
void Root::identify(int fd) {
  set_io_timeout_ms(fd, 2000);  // the message has started arriving; the rest must follow promptly
  Msg first;
  if (!recv_msg(fd, first)) {
    close_fd(fd);
    return;
  }
  if (first.type == MSG_HELLO) {
    const int r = mesh_.hello_rank(first);
    if (r < 0) {
      close_fd(fd);  // not a rank of this rack
    } else if (mesh_.peer_fd(r) < 0) {
      mesh_.adopt_rank(r, fd);
      std::fprintf(stderr, "[rank 0/%d] rank %d joined\n", mesh_.size(), r);
    } else if (!standby_.count(r)) {
      set_io_timeout_ms(fd, 0);  // it may wait a long time
      standby_[r] = fd;
      std::fprintf(stderr, "[rank 0/%d] standby for rank %d is waiting\n", mesh_.size(), r);
    } else {
      close_fd(fd);  // that rank is healthy and already has a standby
    }
    return;
  }
  set_io_timeout_ms(fd, 0);  // clients may be idle for as long as they like
  pending_.push_back(Pending{fd, std::move(first)});
}

// Waits up to timeout_ms (-1 = forever) for a new connection or for a
// newcomer's first message, and handles one such event. False on timeout.
// Newcomers are never read from until they have something to say, so a
// connection that stays silent cannot hold the root up.
bool Root::pump_newcomers(int timeout_ms) {
  std::vector<int> fds = {listen_fd_};
  fds.insert(fds.end(), newcomers_.begin(), newcomers_.end());
  const int i = wait_readable(fds, timeout_ms);
  if (i < 0) return false;
  if (i == 0) {
    const int fd = accept_one(listen_fd_);
    if (fd >= 0) newcomers_.push_back(fd);
  } else {
    const int fd = fds[static_cast<size_t>(i)];
    newcomers_.erase(newcomers_.begin() + (i - 1));
    identify(fd);
  }
  return true;
}

void Root::serve_client_message(const Msg& req) {
  const Msg reply = run(req);
  if (reply.type != MSG_ERROR && req.payload.size() >= sizeof(uint64_t)) {
    uint64_t id;
    std::memcpy(&id, req.payload.data(), sizeof id);
    if (req.type == MSG_PREFILL) owned_.push_back(id);
    if (req.type == MSG_RELEASE) owned_.erase(std::remove(owned_.begin(), owned_.end(), id), owned_.end());
  }
  if (!send_msg(client_, reply)) drop_client();
}

// The client is gone. Free anything it left behind, on every rank, so a
// crashed or careless client cannot leak cache memory.
void Root::drop_client() {
  close_fd(client_);
  client_ = -1;
  const std::vector<uint64_t> left = owned_;
  owned_.clear();
  for (uint64_t id : left) {
    Msg release;
    release.type = MSG_RELEASE;
    release.request_id = next_internal_id_++;
    release.payload.resize(sizeof id);
    std::memcpy(release.payload.data(), &id, sizeof id);
    run(release);
  }
}

int Root::serve() {
  listen_fd_ = listen_on(opt_.port);
  mesh_.set_peer_timeout_ms(opt_.rank_timeout_ms);
  while (!mesh_.missing().empty()) pump_newcomers(-1);  // start-up: wait until every rank has joined
  std::fprintf(stderr,
               "[rank 0/%d] ready on port %d  heads %d-%d  units %d-%d  vocab %d-%d  weights=%.1f MB  recovery=%s\n",
               mesh_.size(), opt_.port, shard_.head_begin, shard_.head_end, shard_.unit_begin, shard_.unit_end,
               shard_.vocab_begin, shard_.vocab_end, shard_.resident_bytes / 1e6,
               opt_.checkpoint ? "checkpoint" : "replay");

  // One loop watches everything: new connections, the current client, and the
  // sockets of the other ranks. A rank's socket should be silent while no
  // request is running, so if one becomes "readable" the rank has gone away,
  // and we learn that at once instead of on the next request.
  for (;;) {
    if (client_ < 0 && !pending_.empty()) {
      Pending p = std::move(pending_.front());
      pending_.pop_front();
      client_ = p.fd;
      serve_client_message(p.first);
      continue;
    }

    std::vector<int> fds;
    std::vector<int> who;  // -2 the client, 1..N-1 that rank, kStandby + r that rank's standby
    if (client_ >= 0) {
      fds.push_back(client_);
      who.push_back(-2);
    }
    for (int r = 1; r < mesh_.size(); ++r) {
      if (mesh_.peer_fd(r) >= 0) {
        fds.push_back(mesh_.peer_fd(r));
        who.push_back(r);
      }
    }
    for (const auto& [r, fd] : standby_) {
      fds.push_back(fd);
      who.push_back(kStandby + r);
    }
    const size_t watched = fds.size();
    fds.push_back(listen_fd_);
    fds.insert(fds.end(), newcomers_.begin(), newcomers_.end());
    const int i = wait_readable(fds, -1);
    if (i < 0) continue;

    if (static_cast<size_t>(i) >= watched) {
      // A new connection, or a newcomer's first message.
      pump_newcomers(0);
      // If that was the replacement we were waiting for, repair now, while
      // nobody is waiting on us.
      if (needs_heal_ && !broken_ && mesh_.missing().empty()) heal(0);
      continue;
    }

    if (who[i] >= kStandby) {  // a standby never speaks, so this means it went away
      const int r = who[i] - kStandby;
      std::fprintf(stderr, "[rank 0/%d] standby for rank %d went away\n", mesh_.size(), r);
      close_fd(standby_[r]);
      standby_.erase(r);
      continue;
    }

    if (who[i] == -2) {
      Msg req;
      if (recv_msg(client_, req)) {
        serve_client_message(req);
      } else {
        drop_client();
      }
    } else {
      std::fprintf(stderr, "[rank 0/%d] failure rank=%d reason=closed during=idle blocked_ms=0.0\n", mesh_.size(), who[i]);
      mesh_.mark_dead(who[i]);
      if (!needs_heal_) failed_at_ = Clock::now();
      needs_heal_ = true;
      if (!broken_) heal(0);  // repairs at once if a standby is waiting; otherwise returns and we wait
    }
  }
}

// ================================================================= requests

// Turns a client message into a request (info + embedded input). Everything
// that can be wrong with a request is checked HERE, before any other rank
// hears about it. Throws on a bad request.
void Root::prepare(const Msg& req, ReqInfo& info, Mat& h, std::vector<int32_t>& tokens) {
  info.request_id = req.request_id;
  info.flags = req.flags & (FLAG_LAST_ONLY | FLAG_ARGMAX);  // FLAG_BACKUP is ours to set, not the client's
  const std::vector<uint8_t>& p = req.payload;
  auto read_session = [&]() {
    if (p.size() < sizeof(uint64_t)) throw std::runtime_error("payload too short");
    std::memcpy(&info.session, p.data(), sizeof(uint64_t));
  };
  switch (req.type) {
    case MSG_TOKENS:
      info.op = OP_FORWARD;
      h = embed_tokens(shard_.cfg, shard_.wte, shard_.wpe, decode_tokens(p));
      break;
    case MSG_PREFILL:
      read_session();
      if (sessions_.find(info.session)) throw std::runtime_error("session already exists");
      if (sessions_.size() >= SessionTable::kMaxSessions) throw std::runtime_error("too many open sessions");
      info.op = OP_PREFILL;
      tokens = decode_tokens(std::vector<uint8_t>(p.begin() + sizeof(uint64_t), p.end()));
      h = embed_tokens(shard_.cfg, shard_.wte, shard_.wpe, tokens);
      break;
    case MSG_DECODE: {
      read_session();
      int32_t token;
      if (p.size() != sizeof(uint64_t) + sizeof token) throw std::runtime_error("bad decode payload");
      std::memcpy(&token, p.data() + sizeof(uint64_t), sizeof token);
      const Session* s = sessions_.find(info.session);
      if (!s) throw std::runtime_error("no such session");
      if (s->len >= static_cast<int>(shard_.cfg.n_ctx)) throw std::runtime_error("the context window is full");
      info.op = OP_DECODE;
      info.pos = static_cast<uint32_t>(s->len);
      tokens = {token};
      h = embed_tokens(shard_.cfg, shard_.wte, shard_.wpe, tokens, s->len);  // position = current length
      break;
    }
    case MSG_RELEASE:
      read_session();
      if (!sessions_.find(info.session)) throw std::runtime_error("no such session");
      info.op = OP_RELEASE;
      break;
    default:
      throw std::runtime_error("unexpected message type");
  }
  if (opt_.checkpoint && (info.op == OP_PREFILL || info.op == OP_DECODE)) info.flags |= FLAG_BACKUP;
}

// Runs one request on the whole rack, exactly once. Throws RankLost if a rank
// fails part-way; in that case nothing has been recorded.
Msg Root::run_once(const Msg& req) {
  ReqInfo info;
  Mat h;
  std::vector<int32_t> tokens;
  try {
    prepare(req, info, h, tokens);
  } catch (const std::exception& e) {
    throw BadRequest(e.what());
  }
  const int T = h.rows;
  const auto t0 = Clock::now();
  mesh_.reset_stats();
  mesh_.start_request(info, h);
  Output out;
  TpStats stats;
  execute(shard_, mesh_, sessions_, info, h, out, stats);

  // Commit. Every rank finished, so only now does the request count as done.
  if (info.op == OP_PREFILL) history_[info.session] = tokens;
  if (info.op == OP_DECODE) history_[info.session].push_back(tokens[0]);
  if (info.op == OP_RELEASE) {
    history_.erase(info.session);
    backups_.erase(info.session);
  }
  if (info.flags & FLAG_BACKUP) {
    std::vector<Mat>& saved = backups_[info.session];
    saved.resize(static_cast<size_t>(mesh_.size()));
    for (int r = 1; r < mesh_.size(); ++r) {
      const Mat& rows = out.backup_parts[static_cast<size_t>(r)];
      Mat& dst = saved[static_cast<size_t>(r)];
      dst.cols = rows.cols;
      dst.rows += rows.rows;
      dst.data.insert(dst.data.end(), rows.data.begin(), rows.data.end());
    }
  }

  Msg reply;
  reply.request_id = req.request_id;
  reply.flags = req.flags;
  if (info.op == OP_RELEASE) {
    reply.type = MSG_OK;
  } else if (info.flags & FLAG_ARGMAX) {
    reply.type = MSG_TOKEN;
    reply.payload.resize(sizeof out.token + sizeof out.token_logit);
    std::memcpy(reply.payload.data(), &out.token, sizeof out.token);
    std::memcpy(reply.payload.data() + sizeof out.token, &out.token_logit, sizeof out.token_logit);
  } else {
    reply.type = MSG_LOGITS;
    reply.payload = encode_mat(out.logits);
  }
  log_request(mesh_, info, T, ms_since(t0), stats, out);
  return reply;
}

// Runs a request, repairing the rack and trying again if a rank fails.
Msg Root::run(const Msg& req) {
  if (req.type == MSG_PING) {
    Msg pong;
    pong.type = MSG_PONG;
    pong.request_id = req.request_id;
    return pong;
  }
  if (broken_) return make_error(req.request_id, "the rack is down after an unrecoverable error; restart it");

  for (int attempt = 0; attempt < 4; ++attempt) {
    if (needs_heal_) {
      const Heal result = heal(opt_.rejoin_timeout_ms);
      if (result == Heal::still_missing) {
        std::string who;
        for (int r : mesh_.missing()) who += (who.empty() ? "" : ", ") + std::to_string(r);
        return make_error(req.request_id, "the rack is degraded: rank " + who + " was lost and no replacement joined within " +
                                              std::to_string(opt_.rejoin_timeout_ms) + " ms; nothing was changed, try again");
      }
      if (result == Heal::failed_again) continue;
    }
    const auto t0 = Clock::now();
    try {
      return run_once(req);
    } catch (const BadRequest& e) {
      return make_error(req.request_id, e.what());  // nothing was started; the rack is untouched
    } catch (const RankLost& e) {
      on_rank_lost(e, "request", ms_since(t0));  // then loop: heal, and run the request again
    } catch (const std::exception& e) {
      // For example the replica check failing. We do not know which rank is
      // wrong, so we stop rather than risk serving wrong answers.
      broken_ = true;
      std::fprintf(stderr, "[rank 0/%d] rack failed during req %llu: %s\n", mesh_.size(),
                   static_cast<unsigned long long>(req.request_id), e.what());
      return make_error(req.request_id, e.what());
    }
  }
  return make_error(req.request_id, "giving up: the rack failed repeatedly while running this request");
}

void Root::start_internal(ReqInfo& info, const Mat& h) {
  info.request_id = next_internal_id_++;
  mesh_.start_request(info, h);
}

// ================================================================= failures

void Root::on_rank_lost(const RankLost& e, const char* during, double blocked_ms) {
  std::fprintf(stderr, "[rank 0/%d] failure rank=%d reason=%s during=%s blocked_ms=%.1f\n", mesh_.size(), e.rank,
               e.timed_out ? "timeout" : "closed", during, blocked_ms);
  if (!needs_heal_) failed_at_ = Clock::now();
  // Closing the socket also fences the old process: if it was only hung and
  // wakes up later, it finds its connection gone and exits, so it can never
  // take part with stale state.
  mesh_.mark_dead(e.rank);
  mesh_.abort_survivors();
  needs_heal_ = true;
}

Root::Heal Root::heal(int wait_ms) {
  // 1. Get a replacement for every missing rank: a standby if one is waiting,
  //    otherwise wait for a new process to connect.
  const auto t_wait = Clock::now();
  int from_standby = 0;
  for (int r : mesh_.missing()) {
    auto it = standby_.find(r);
    if (it == standby_.end()) continue;
    mesh_.adopt_rank(r, it->second);
    standby_.erase(it);
    ++from_standby;
    std::fprintf(stderr, "[rank 0/%d] rank %d replaced by its standby\n", mesh_.size(), r);
  }
  while (!mesh_.missing().empty()) {
    const int left = wait_ms - static_cast<int>(ms_since(t_wait));
    if (left <= 0 || !pump_newcomers(left)) return Heal::still_missing;
  }
  const double wait_rejoin_ms = ms_since(t_wait);
  const auto t_rebuild = Clock::now();
  uint64_t tokens = 0, restored_bytes = 0;

  try {
    // 2. Put my own table back to the last committed point: drop sessions that
    //    were never committed, and cut off rows a half-finished step added.
    for (uint64_t id : sessions_.ids()) {
      auto it = history_.find(id);
      if (it == history_.end()) {
        sessions_.erase(id);
      } else {
        sessions_.find(id)->truncate(static_cast<int>(it->second.size()));
      }
    }

    // 3. Every other rank starts from empty: survivors may hold half a step,
    //    and the replacement holds nothing.
    ReqInfo reset;
    reset.op = OP_RESET;
    start_internal(reset, Mat());

    // 4. Rebuild.
    if (opt_.checkpoint) {
      // Hand each rank back the cache rows it had sent us.
      for (const auto& [id, toks] : history_) {
        const std::vector<Mat>& saved = backups_.at(id);
        for (int r = 1; r < mesh_.size(); ++r) {
          ReqInfo info;
          info.op = OP_RESTORE;
          info.session = id;
          info.pos = static_cast<uint32_t>(toks.size());
          info.target = static_cast<uint32_t>(r);
          start_internal(info, Mat());
          mesh_.send_to_rank(r, saved[static_cast<size_t>(r)]);
          restored_bytes += saved[static_cast<size_t>(r)].size() * sizeof(float);
        }
        tokens += toks.size();
      }
    } else {
      // Re-run every session's tokens as one prefill. Part 4 showed prefill
      // and step-by-step decode give bit-identical caches, so what is rebuilt
      // is exactly what was lost.
      sessions_.clear();
      for (const auto& [id, toks] : history_) {
        ReqInfo info;
        info.op = OP_PREFILL;
        info.flags = FLAG_ARGMAX;  // we do not need the logits; keep the reply tiny
        info.session = id;
        Mat h = embed_tokens(shard_.cfg, shard_.wte, shard_.wpe, toks);
        start_internal(info, h);
        Output out;
        TpStats stats;
        execute(shard_, mesh_, sessions_, info, h, out, stats);
        tokens += toks.size();
      }
    }
  } catch (const RankLost& e) {
    on_rank_lost(e, "recovery", 0.0);
    return Heal::failed_again;
  }

  needs_heal_ = false;
  std::fprintf(stderr,
               "[rank 0/%d] recovery strategy=%s standbys_used=%d wait_rejoin_ms=%.1f rebuild_ms=%.1f sessions=%zu "
               "tokens=%llu restored_kb=%.1f total_ms=%.1f\n",
               mesh_.size(), opt_.checkpoint ? "checkpoint" : "replay", from_standby, wait_rejoin_ms, ms_since(t_rebuild),
               history_.size(), static_cast<unsigned long long>(tokens), restored_bytes / 1e3, ms_since(failed_at_));
  return Heal::ok;
}

}  // namespace

int serve_as_root(const RootOptions& opt, const TpShard& shard, Mesh& mesh) {
  Root root(opt, shard, mesh);
  return root.serve();
}

}  // namespace mr
