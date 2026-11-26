// root.h - rank 0: the rank that talks to clients and keeps the rack healthy.
#pragma once

#include "mesh.h"
#include "tp_model.h"

namespace mr {

struct RootOptions {
  int port = 0;
  // How state is rebuilt on a replacement rank (Part 5):
  //   false  replay:     re-run each session's tokens. Costs nothing until a
  //                      failure; rebuild time grows with the session length.
  //   true   checkpoint: after every step each rank sends the root a copy of
  //                      its new cache rows; a replacement is handed them
  //                      back. Costs bytes and memory on every step; rebuild
  //                      is a copy.
  bool checkpoint = false;
  int rank_timeout_ms = 10000;    // a rank silent for this long is treated as dead
  int rejoin_timeout_ms = 30000;  // how long a request waits for a replacement rank
};

// Runs the root's event loop. Does not return in normal operation.
int serve_as_root(const RootOptions& opt, const TpShard& shard, Mesh& mesh);

}  // namespace mr
