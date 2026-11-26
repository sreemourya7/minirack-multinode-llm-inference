#include "mesh.h"

#include <chrono>
#include <cstring>
#include <stdexcept>
#include <thread>

#include "net.h"

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

}  // namespace

Mesh::Mesh(int rank, int nranks) : rank_(rank), nranks_(nranks), fds_(static_cast<size_t>(nranks), -1) {
  if (nranks < 1 || rank < 0 || rank >= nranks) throw std::runtime_error("bad rank / nranks");
}

Mesh::~Mesh() {
  for (int fd : fds_) close_fd(fd);
}

void Mesh::accept_peers(int listen_fd) {
  int missing = nranks_ - 1;
  while (missing > 0) {
    const int fd = accept_one(listen_fd);
    if (fd < 0) continue;
    Msg hello;
    uint32_t who[2] = {0, 0};
    const bool ok = recv_msg(fd, hello) && hello.type == MSG_HELLO && hello.payload.size() == sizeof who;
    if (ok) std::memcpy(who, hello.payload.data(), sizeof who);
    // Anything that is not a valid, new rank (for example a client that
    // connected too early) is turned away; it can retry once the mesh is up.
    if (!ok || who[1] != static_cast<uint32_t>(nranks_) || who[0] == 0 || who[0] >= static_cast<uint32_t>(nranks_) ||
        fds_[who[0]] >= 0) {
      close_fd(fd);
      continue;
    }
    fds_[who[0]] = fd;
    --missing;
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
  const uint32_t who[2] = {static_cast<uint32_t>(rank_), static_cast<uint32_t>(nranks_)};
  hello.payload.resize(sizeof who);
  std::memcpy(hello.payload.data(), who, sizeof who);
  if (!send_msg(fd, hello)) throw std::runtime_error("lost the root while saying hello");
  fds_[0] = fd;
}

// Every message inside a request carries (request_id, seq). The receiver
// checks both. If two ranks ever disagree about which step they are on, we
// find out at once instead of silently adding the wrong matrices together.
void Mesh::send_mat(int fd, const Mat& m) {
  Msg msg;
  msg.type = MSG_COLL;
  msg.request_id = request_id_;
  msg.seq = seq_;
  msg.payload = encode_mat(m);
  if (!send_msg(fd, msg)) throw std::runtime_error("collective: lost a rank while sending");
  stats_.bytes_sent += msg.payload.size();
}

Mat Mesh::recv_mat(int fd) {
  Msg msg;
  if (!recv_msg(fd, msg)) throw std::runtime_error("collective: lost a rank while receiving");
  if (msg.type != MSG_COLL || msg.request_id != request_id_ || msg.seq != seq_) {
    throw std::runtime_error("collective: ranks are out of step");
  }
  stats_.bytes_recv += msg.payload.size();
  return decode_mat(msg.payload);
}

void Mesh::start_request(uint64_t request_id, uint32_t flags, const Mat& h0) {
  if (!is_root()) throw std::runtime_error("only the root starts requests");
  request_id_ = request_id;
  seq_ = 0;
  if (nranks_ == 1) return;  // nobody to tell
  ScopedTimer timer(stats_);
  Msg msg;
  msg.type = MSG_START;
  msg.flags = flags;
  msg.request_id = request_id;
  msg.payload = encode_mat(h0);
  for (int r = 1; r < nranks_; ++r) {
    if (!send_msg(fds_[r], msg)) throw std::runtime_error("lost a rank while starting a request");
    stats_.bytes_sent += msg.payload.size();
  }
}

bool Mesh::wait_request(uint64_t& request_id, uint32_t& flags, Mat& h0) {
  Msg msg;
  if (!recv_msg(fds_[0], msg) || msg.type != MSG_START) return false;
  request_id_ = request_id = msg.request_id;
  flags = msg.flags;
  seq_ = 0;
  h0 = decode_mat(msg.payload);
  stats_.bytes_recv += msg.payload.size();
  return true;
}

void Mesh::all_reduce_sum(Mat& m) {
  ++seq_;
  if (nranks_ == 1) return;  // nothing to combine with
  ScopedTimer timer(stats_);
  if (is_root()) {
    // Reduce: add every rank's part to mine, in rank order.
    for (int r = 1; r < nranks_; ++r) {
      const Mat part = recv_mat(fds_[r]);
      if (part.rows != m.rows || part.cols != m.cols) throw std::runtime_error("all_reduce: shape mismatch between ranks");
      for (size_t i = 0; i < m.data.size(); ++i) m.data[i] += part.data[i];
    }
    // Broadcast: give everyone the total.
    for (int r = 1; r < nranks_; ++r) send_mat(fds_[r], m);
  } else {
    send_mat(fds_[0], m);
    m = recv_mat(fds_[0]);
  }
}

void Mesh::gather_cols(const Mat& local, Mat& full) {
  ++seq_;
  if (!is_root()) {
    ScopedTimer timer(stats_);
    send_mat(fds_[0], local);
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
    parts.push_back(recv_mat(fds_[r]));
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

void Mesh::check_identical(const Mat& m) {
  ++seq_;
  if (nranks_ == 1) return;
  ScopedTimer timer(stats_);
  Mat sum(1, 2);  // a 64-bit checksum carried as two 32-bit halves in a 1x2 matrix
  const uint64_t mine = checksum(m);
  std::memcpy(sum.data.data(), &mine, sizeof mine);
  if (!is_root()) {
    send_mat(fds_[0], sum);
    return;
  }
  for (int r = 1; r < nranks_; ++r) {
    const Mat theirs = recv_mat(fds_[r]);
    if (theirs.size() != 2 || std::memcmp(theirs.data.data(), &mine, sizeof mine) != 0) {
      throw std::runtime_error("replica check failed: rank " + std::to_string(r) + " holds a different hidden state");
    }
  }
}

}  // namespace mr
