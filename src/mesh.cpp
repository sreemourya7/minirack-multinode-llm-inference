#include "mesh.h"

#include "kernels.h"

#include <chrono>
#include <cstring>
#include <stdexcept>
#include <thread>

namespace mr {

namespace {

// Adds the time spent in its scope to a CommStats.
struct ScopedTimer {
  CommStats& s;
  std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
  explicit ScopedTimer(CommStats& stats) : s(stats) { ++s.calls; }
  ~ScopedTimer() { s.seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); }
};

// FNV-1a: a tiny, fast checksum. Not cryptographic; it only has to notice
// that two byte strings differ.
uint64_t checksum(const Mat& m) {
  uint64_t h = 1469598103934665603ull;
  const auto* p = reinterpret_cast<const uint8_t*>(m.data.data());
  const size_t n = m.size() * sizeof(float);
  for (size_t i = 0; i < n; ++i) {
    h ^= p[i];
    h *= 1099511628211ull;
  }
  return h;
}

// Fixed-size part of a MSG_START payload; the input matrix follows it.
struct StartPrefix {
  uint32_t op;
  uint32_t pos;
  uint64_t session;
  uint32_t target;
  uint32_t reserved;
};
static_assert(sizeof(StartPrefix) == 24, "StartPrefix must be 24 bytes");

}  // namespace

Mesh::Mesh(int rank, int nranks) : rank_(rank), nranks_(nranks), fds_(static_cast<size_t>(nranks), -1) {
  if (nranks < 1 || rank < 0 || rank >= nranks) throw std::runtime_error("bad rank / nranks");
}

Mesh::~Mesh() {
  for (int fd : fds_) close_fd(fd);
}

// ------------------------------------------------------------------ membership

int Mesh::hello_rank(const Msg& hello) const {
  uint32_t who[3] = {0, 0, 0};  // rank, rack size, kernel implementation
  if (hello.type != MSG_HELLO || hello.payload.size() != sizeof who) return -1;
  std::memcpy(who, hello.payload.data(), sizeof who);
  const bool valid = who[1] == static_cast<uint32_t>(nranks_) && who[0] >= 1 && who[0] < static_cast<uint32_t>(nranks_);
  if (!valid) return -1;  // wrong rack size or impossible rank
  // Every rank must do its arithmetic the same way as the root. Otherwise a
  // replacement could not reproduce exactly what the rank it replaces computed.
  if (who[2] != static_cast<uint32_t>(kern::active())) return -2;
  return static_cast<int>(who[0]);
}

void Mesh::adopt_rank(int r, int fd) {
  if (fds_[static_cast<size_t>(r)] >= 0) throw std::runtime_error("adopt_rank: that rank is not missing");
  set_io_timeout_ms(fd, peer_timeout_ms_);
  fds_[static_cast<size_t>(r)] = fd;
}

std::vector<int> Mesh::missing() const {
  std::vector<int> out;
  for (int r = 1; r < nranks_; ++r) {
    if (fds_[static_cast<size_t>(r)] < 0) out.push_back(r);
  }
  return out;
}

void Mesh::mark_dead(int r) {
  close_fd(fds_[static_cast<size_t>(r)]);
  fds_[static_cast<size_t>(r)] = -1;
}

void Mesh::abort_survivors() {
  // Each surviving rank may be anywhere: computing, about to send us a
  // matrix, or waiting for one from us. So: send ABORT, then read and throw
  // away whatever it had already sent, until its ABORT_ACK arrives. After the
  // ack the socket is known to be empty in both directions.
  Msg abort_msg;
  abort_msg.type = MSG_ABORT;
  for (int r = 1; r < nranks_; ++r) {
    const int fd = fds_[static_cast<size_t>(r)];
    if (fd < 0) continue;
    bool acked = false;
    if (send_msg(fd, abort_msg)) {
      Msg m;
      while (recv_msg(fd, m)) {
        if (m.type == MSG_ABORT_ACK) {
          acked = true;
          break;
        }
      }
    }
    if (!acked) mark_dead(r);  // it died too; the caller will see it in missing()
  }
}

void Mesh::connect_to_root(const std::string& host, int port) {
  int fd = -1;
  for (int attempt = 0; attempt < 300 && fd < 0; ++attempt) {  // up to ~30 s
    fd = connect_to(host, port);
    if (fd < 0) std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
  if (fd < 0) throw std::runtime_error("cannot reach the root rank");
  Msg hello;
  hello.type = MSG_HELLO;
  const uint32_t who[3] = {static_cast<uint32_t>(rank_), static_cast<uint32_t>(nranks_),
                           static_cast<uint32_t>(kern::active())};
  hello.payload.resize(sizeof who);
  std::memcpy(hello.payload.data(), who, sizeof who);
  if (!send_msg(fd, hello)) throw std::runtime_error("lost the root while saying hello");
  fds_[0] = fd;
}

// ------------------------------------------------------------ sending / receiving

void Mesh::peer_failed(int peer) {
  if (is_root()) throw RankLost(peer, last_io_timed_out());
  throw std::runtime_error("lost the connection to the root");
}

// Every message inside a request carries (request_id, seq). The receiver
// checks both. If two ranks ever disagree about which step they are on, we
// find out at once instead of silently adding the wrong matrices together.
void Mesh::send_mat(int peer, const Mat& m) {
  Msg msg;
  msg.type = MSG_COLL;
  msg.request_id = request_id_;
  msg.seq = seq_;
  msg.payload = encode_mat(m);
  if (!send_msg(fds_[static_cast<size_t>(peer)], msg)) peer_failed(peer);
  stats_.bytes_sent += msg.payload.size();
}

Mat Mesh::recv_mat(int peer) {
  const int fd = fds_[static_cast<size_t>(peer)];
  Msg msg;
  if (!recv_msg(fd, msg)) peer_failed(peer);
  if (!is_root() && msg.type == MSG_ABORT) {
    // The root gave up on this request (another rank failed). Acknowledge, so
    // the root knows this socket is clean, and unwind to the request loop.
    Msg ack;
    ack.type = MSG_ABORT_ACK;
    send_msg(fd, ack);
    throw RequestAborted();
  }
  if (msg.type != MSG_COLL || msg.request_id != request_id_ || msg.seq != seq_) {
    throw std::runtime_error("collective: ranks are out of step");
  }
  stats_.bytes_recv += msg.payload.size();
  return decode_mat(msg.payload);
}

// ------------------------------------------------------------ request boundaries

void Mesh::start_request(const ReqInfo& info, const Mat& h0) {
  if (!is_root()) throw std::runtime_error("only the root starts requests");
  request_id_ = info.request_id;
  seq_ = 0;
  if (nranks_ == 1) return;  // nobody to tell
  ScopedTimer timer(stats_);
  Msg msg;
  msg.type = MSG_START;
  msg.flags = info.flags;
  msg.request_id = info.request_id;
  const StartPrefix prefix{info.op, info.pos, info.session, info.target, 0};
  const std::vector<uint8_t> mat = encode_mat(h0);
  msg.payload.resize(sizeof prefix + mat.size());
  std::memcpy(msg.payload.data(), &prefix, sizeof prefix);
  std::memcpy(msg.payload.data() + sizeof prefix, mat.data(), mat.size());
  for (int r = 1; r < nranks_; ++r) {
    if (!send_msg(fds_[static_cast<size_t>(r)], msg)) peer_failed(r);
    stats_.bytes_sent += msg.payload.size();
  }
}

bool Mesh::wait_request(ReqInfo& info, Mat& h0) {
  for (;;) {
    Msg msg;
    if (!recv_msg(fds_[0], msg)) return false;
    if (msg.type == MSG_ABORT) {
      // We had already finished our side of the request the root is
      // cancelling. Nothing to undo here; just acknowledge and keep waiting.
      Msg ack;
      ack.type = MSG_ABORT_ACK;
      if (!send_msg(fds_[0], ack)) return false;
      continue;
    }
    StartPrefix prefix;
    if (msg.type != MSG_START || msg.payload.size() < sizeof prefix) return false;
    std::memcpy(&prefix, msg.payload.data(), sizeof prefix);
    request_id_ = info.request_id = msg.request_id;
    info.flags = msg.flags;
    info.op = prefix.op;
    info.pos = prefix.pos;
    info.session = prefix.session;
    info.target = prefix.target;
    seq_ = 0;
    h0 = decode_mat(std::vector<uint8_t>(msg.payload.begin() + sizeof prefix, msg.payload.end()));
    stats_.bytes_recv += msg.payload.size();
    return true;
  }
}

// ----------------------------------------------------------------- collectives

void Mesh::all_reduce_sum(Mat& m) {
  ++seq_;
  if (nranks_ == 1) return;  // nothing to combine with
  ScopedTimer timer(stats_);
  if (is_root()) {
    // Reduce: add every rank's part to mine, in rank order.
    for (int r = 1; r < nranks_; ++r) {
      const Mat part = recv_mat(r);
      if (part.rows != m.rows || part.cols != m.cols) throw std::runtime_error("all_reduce: shape mismatch between ranks");
      for (size_t i = 0; i < m.data.size(); ++i) m.data[i] += part.data[i];
    }
    // Broadcast: give everyone the total.
    for (int r = 1; r < nranks_; ++r) send_mat(r, m);
  } else {
    send_mat(0, m);
    m = recv_mat(0);
  }
}

void Mesh::gather_cols(const Mat& local, Mat& full) {
  ++seq_;
  if (!is_root()) {
    ScopedTimer timer(stats_);
    send_mat(0, local);
    return;
  }
  if (nranks_ == 1) {
    full = local;
    return;
  }
  ScopedTimer timer(stats_);
  std::vector<Mat> parts;
  int cols = local.cols;
  for (int r = 1; r < nranks_; ++r) {
    parts.push_back(recv_mat(r));
    if (parts.back().rows != local.rows) throw std::runtime_error("gather: row count differs between ranks");
    cols += parts.back().cols;
  }
  full = Mat(local.rows, cols);
  for (int t = 0; t < local.rows; ++t) {
    float* dst = full.row(t);
    std::memcpy(dst, local.row(t), static_cast<size_t>(local.cols) * sizeof(float));
    dst += local.cols;
    for (const Mat& p : parts) {
      std::memcpy(dst, p.row(t), static_cast<size_t>(p.cols) * sizeof(float));
      dst += p.cols;
    }
  }
}

void Mesh::gather_parts(const Mat& local, std::vector<Mat>& parts) {
  ++seq_;
  if (nranks_ == 1) return;
  ScopedTimer timer(stats_);
  if (!is_root()) {
    send_mat(0, local);
    return;
  }
  parts.assign(static_cast<size_t>(nranks_), Mat());
  for (int r = 1; r < nranks_; ++r) parts[static_cast<size_t>(r)] = recv_mat(r);
}

void Mesh::check_identical(const Mat& m) {
  ++seq_;
  if (nranks_ == 1) return;
  ScopedTimer timer(stats_);
  Mat sum(1, 2);  // a 64-bit checksum carried as two 32-bit halves in a 1x2 matrix
  const uint64_t mine = checksum(m);
  std::memcpy(sum.data.data(), &mine, sizeof mine);
  if (!is_root()) {
    send_mat(0, sum);
    return;
  }
  for (int r = 1; r < nranks_; ++r) {
    const Mat theirs = recv_mat(r);
    if (theirs.size() != 2 || std::memcmp(theirs.data.data(), &mine, sizeof mine) != 0) {
      throw std::runtime_error("replica check failed: rank " + std::to_string(r) + " holds a different hidden state");
    }
  }
}

void Mesh::send_to_rank(int r, const Mat& m) {
  ++seq_;
  ScopedTimer timer(stats_);
  send_mat(r, m);
}

Mat Mesh::recv_from_root() {
  ++seq_;
  ScopedTimer timer(stats_);
  return recv_mat(0);
}

}  // namespace mr
