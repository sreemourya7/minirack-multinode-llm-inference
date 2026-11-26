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
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "mat.h"

namespace mr {

// What a request asks the rack to do (Part 4 added everything but OP_FORWARD).
enum Op : uint32_t {
  OP_FORWARD = 0,  // stateless: run a whole sequence, keep nothing
  OP_PREFILL = 1,  // create a session and fill its KV cache from a prompt
  OP_DECODE = 2,   // extend a session by one token
  OP_RELEASE = 3,  // free a session
};

// Sent by the root to every rank at the start of a request.
struct ReqInfo {
  uint64_t request_id = 0;
  uint32_t flags = 0;
  uint32_t op = OP_FORWARD;
  uint64_t session = 0;
  uint32_t pos = 0;  // decode: the session length the root expects every rank to have
};

struct CommStats {
  uint64_t calls = 0;       // number of collectives
  uint64_t bytes_sent = 0;  // payload bytes
  uint64_t bytes_recv = 0;
  double seconds = 0.0;     // time inside collectives: copying AND waiting for other ranks
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

  // --- setup -----------------------------------------------------------------
  // Root: wait on listen_fd until every other rank has connected and said hello.
  void accept_peers(int listen_fd);
  // Other ranks: connect to the root (retrying while it starts up) and say hello.
  void connect_to_root(const std::string& host, int port);

  // --- request boundaries ----------------------------------------------------
  // Root only: start a request on every rank, handing each the embedded input.
  void start_request(const ReqInfo& info, const Mat& h0);
  // Other ranks: block until the root starts a request. False if the root is gone.
  bool wait_request(ReqInfo& info, Mat& h0);

  // --- collectives (every rank must call the same ones in the same order) -----
  void all_reduce_sum(Mat& m);                     // in: my part.  out: the sum
  void gather_cols(const Mat& local, Mat& full);   // `full` is filled on the root only
  void check_identical(const Mat& m);              // throws on the root if copies differ

  const CommStats& stats() const { return stats_; }
  void reset_stats() { stats_ = CommStats{}; }

 private:
  void send_mat(int fd, const Mat& m);
  Mat recv_mat(int fd);

  int rank_;
  int nranks_;
  std::vector<int> fds_;     // root: fds_[r] is the socket to rank r.  others: fds_[0] is the root.
  uint64_t request_id_ = 0;
  uint32_t seq_ = 0;         // which collective of this request we are on
  CommStats stats_;
};

}  // namespace mr
