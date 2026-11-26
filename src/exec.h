// exec.h - the program every rank runs for one request.
//
// The root (root.cpp) and the other ranks (rank.cpp) both call execute(). All
// ranks take the same branches, so they issue the same collectives in the
// same order. That shared shape is what keeps them in step.
#pragma once

#include <chrono>
#include <cstdint>
#include <vector>

#include "kv_cache.h"
#include "mat.h"
#include "mesh.h"
#include "tp_model.h"

namespace mr {

// What a request produced. Only the root fills logits / token / backup_parts.
struct Output {
  Mat logits;
  int32_t token = -1;
  float token_logit = 0.0f;
  int session_len = 0;            // tokens in the session's KV cache afterwards
  uint64_t kv_bytes = 0;          // size of THIS rank's part of that cache
  double head_ms = 0.0;           // time this rank spent on the vocabulary projection
  std::vector<Mat> backup_parts;  // with FLAG_BACKUP: backup_parts[r] = the cache rows rank r just added
};

const char* op_name(uint32_t op);
double ms_since(std::chrono::steady_clock::time_point t0);

void execute(const TpShard& shard, Mesh& mesh, SessionTable& sessions, const ReqInfo& info, Mat& h, Output& out,
             TpStats& stats);

// One line per request on stderr, in key=value form so the tests can read it.
void log_request(const Mesh& mesh, const ReqInfo& info, int T, double total_ms, const TpStats& st, const Output& out);

}  // namespace mr
