// mesh.h - collectives: the operations a group of ranks performs together.
//
// In Part 2 every rank runs the SAME program on its OWN slice of the weights.
// Most lines are local work. A few lines are "collectives": points where all
// ranks stop, exchange data, and continue with a shared result.
//
//   start_request   root tells every rank what to do and gives it
//                   the same input                                 (a broadcast)
//   all_reduce_sum  every rank puts in a matrix; every rank gets
//                   back the SUM of all of them
//   gather_cols     every rank puts in some columns; the root gets
//                   them all, side by side
//   check_identical every rank puts in a checksum; the root verifies
//                   they are all equal
//
// Wiring: a star. Every rank holds one connection, to the root (rank 0).
//
//   rank 1 ---+
//   rank 2 ---+--- rank 0 (root) --- client
//   rank 3 ---+
//
// all_reduce_sum on a star is "reduce to the root, then broadcast":
//   1. every other rank sends its matrix to the root
//   2. the root adds them up, always in rank order 0 + 1 + 2 + ...
//   3. the root sends the sum back to every rank
// Because the sum is formed in exactly one place and one order, every rank
// ends up with bit-identical numbers. That matters: the ranks each keep their
// own copy of the hidden state, and those copies must never drift apart.
//
// Part 5 adds failure handling. On the root, a rank that closes its
// connection or stays silent past a timeout raises RankLost. The root then
//   mark_dead        closes the lost rank's socket (this also "fences" it: a
//                    rank that was only hung finds its connection gone and exits)
//   abort_survivors  tells the remaining ranks to drop the request in progress
//                    and waits for each to acknowledge, so that no stale
//                    message is left in any socket
//   adopt_rank       installs a replacement process for the missing rank
#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "mat.h"
#include "net.h"

namespace mr {

// What a request asks the rack to do.
enum Op : uint32_t {
  OP_FORWARD = 0,  // stateless: run a whole sequence, keep nothing
  OP_PREFILL = 1,  // create a session and fill its KV cache from a prompt
  OP_DECODE = 2,   // extend a session by one token
  OP_RELEASE = 3,  // free a session
  OP_RESET = 4,    // (Part 5) every non-root rank drops all its sessions
  OP_RESTORE = 5,  // (Part 5) the root hands one rank a saved copy of its cache for a session
};

// Sent by the root to every rank at the start of a request.
struct ReqInfo {
  uint64_t request_id = 0;
  uint32_t flags = 0;
  uint32_t op = OP_FORWARD;
  uint64_t session = 0;
  uint32_t pos = 0;     // decode: the session length the root expects every rank to have
  uint32_t target = 0;  // restore: the rank that receives the saved cache
};

struct CommStats {
  uint64_t calls = 0;       // number of collectives
  uint64_t bytes_sent = 0;  // payload bytes
  uint64_t bytes_recv = 0;
  double seconds = 0.0;     // time inside collectives: copying AND waiting for other ranks
};

// Root only: rank `rank` stopped answering.
struct RankLost : std::runtime_error {
  int rank;
  bool timed_out;  // true: silent for longer than the timeout. false: connection closed.
  RankLost(int r, bool timeout)
      : std::runtime_error("rank " + std::to_string(r) + (timeout ? " timed out" : " closed its connection")),
        rank(r),
        timed_out(timeout) {}
};

// Non-root ranks only: the root cancelled the request in progress.
struct RequestAborted : std::runtime_error {
  RequestAborted() : std::runtime_error("the root aborted this request") {}
};

class Mesh {
 public:
  Mesh(int rank, int nranks);
  ~Mesh();
  Mesh(const Mesh&) = delete;
  Mesh& operator=(const Mesh&) = delete;

  int rank() const { return rank_; }
  int size() const { return nranks_; }
  bool is_root() const { return rank_ == 0; }

  // --- membership (root) -------------------------------------------------------
  // Which rank a HELLO message claims to be; -1 if it is not a valid HELLO for
  // this rack; -2 if it is valid but the sender uses different kernels (Part 6).
  int hello_rank(const Msg& hello) const;
  // Installs fd as the connection to rank r, which must currently be missing.
  void adopt_rank(int r, int fd);
  std::vector<int> missing() const;       // ranks with no connection right now
  int peer_fd(int r) const { return fds_[static_cast<size_t>(r)]; }
  void mark_dead(int r);                  // close and forget rank r's connection
  void abort_survivors();                 // cancel the request in progress on every live rank
  void set_peer_timeout_ms(int ms) { peer_timeout_ms_ = ms; }

  // --- membership (other ranks) --------------------------------------------------
  // Connect to the root (retrying while it starts up) and say hello.
  void connect_to_root(const std::string& host, int port);

  // --- request boundaries ------------------------------------------------------
  // Root only: start a request on every rank, handing each the embedded input.
  void start_request(const ReqInfo& info, const Mat& h0);
  // Other ranks: block until the root starts a request. False if the root is gone.
  bool wait_request(ReqInfo& info, Mat& h0);

  // --- collectives (every rank must call the same ones in the same order) -------
  void all_reduce_sum(Mat& m);                     // in: my part.  out: the sum
  void gather_cols(const Mat& local, Mat& full);   // `full` is filled on the root only
  void check_identical(const Mat& m);              // throws on the root if copies differ
  // Every non-root rank sends a matrix; on the root parts[r] is rank r's
  // (parts[0] is left empty). Used to back up cache rows (Part 5).
  void gather_parts(const Mat& local, std::vector<Mat>& parts);

  // --- point to point, inside a request ------------------------------------------
  void send_to_rank(int r, const Mat& m);  // root only
  Mat recv_from_root();                    // the matching call on rank r

  const CommStats& stats() const { return stats_; }
  void reset_stats() { stats_ = CommStats{}; }

 private:
  void send_mat(int peer, const Mat& m);
  Mat recv_mat(int peer);
  [[noreturn]] void peer_failed(int peer);

  int rank_;
  int nranks_;
  std::vector<int> fds_;     // root: fds_[r] is the socket to rank r, -1 if missing.  others: fds_[0] is the root.
  int peer_timeout_ms_ = 0;  // root: how long a rank may stay silent (0 = forever)
  uint64_t request_id_ = 0;
  uint32_t seq_ = 0;         // which collective of this request we are on
  CommStats stats_;
};

}  // namespace mr
