#include "exec.h"

#include "prof.h"

#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace mr {

const char* op_name(uint32_t op) {
  switch (op) {
    case OP_FORWARD: return "forward";
    case OP_PREFILL: return "prefill";
    case OP_DECODE: return "decode";
    case OP_RELEASE: return "release";
    case OP_RESET: return "reset";
    case OP_RESTORE: return "restore";
    default: return "?";
  }
}

double ms_since(std::chrono::steady_clock::time_point t0) {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

void log_request(const Mesh& mesh, const ReqInfo& info, int T, double total_ms, const TpStats& st, const Output& out) {
  const CommStats& c = mesh.stats();
  std::string load;  // tokens routed to each expert, e.g. "31,12,40,..."
  for (uint64_t n : st.expert_load) load += (load.empty() ? "" : ",") + std::to_string(n);
  std::fprintf(stderr,
               "[rank %d/%d] req %llu op=%s T=%d len=%d kv_kb=%.1f total_ms=%.2f compute_ms=%.2f head_ms=%.2f "
               "matmul_ms=%.2f attention_ms=%.2f gelu_ms=%.2f norm_ms=%.2f "
               "collective_ms=%.2f collectives=%llu sent_kb=%.1f recv_kb=%.1f expert_rows=%llu expert_load=[%s]\n",
               mesh.rank(), mesh.size(), static_cast<unsigned long long>(info.request_id), op_name(info.op), T,
               out.session_len, out.kv_bytes / 1e3, total_ms, total_ms - c.seconds * 1e3, out.head_ms,
               prof::ms[prof::kMatmul], prof::ms[prof::kAttention], prof::ms[prof::kGelu], prof::ms[prof::kNorm],
               c.seconds * 1e3,
               static_cast<unsigned long long>(c.calls), c.bytes_sent / 1e3, c.bytes_recv / 1e3,
               static_cast<unsigned long long>(st.expert_rows), load.c_str());
}

void execute(const TpShard& shard, Mesh& mesh, SessionTable& sessions, const ReqInfo& info, Mat& h, Output& out,
             TpStats& stats) {
  prof::reset();
  const int n_layer = static_cast<int>(shard.cfg.n_layer);
  const int my_width = (shard.head_end - shard.head_begin) * static_cast<int>(shard.cfg.n_embd / shard.cfg.n_head);

  // --- requests that only manage state -----------------------------------------
  if (info.op == OP_RELEASE) {
    if (!sessions.erase(info.session)) throw std::runtime_error("release: this rank has no such session");
    return;
  }
  if (info.op == OP_RESET) {
    // Recovery, step 1: the other ranks forget every session. (The root has
    // already put its own table in order.)
    if (!mesh.is_root()) sessions.clear();
    return;
  }
  if (info.op == OP_RESTORE) {
    // Recovery from a checkpoint: the root sends the target rank the cache
    // rows it had saved for it. The root's side of this is in root.cpp.
    if (mesh.rank() == static_cast<int>(info.target)) {
      const Mat rows = mesh.recv_from_root();
      Session& s = sessions.create(info.session, n_layer, my_width);
      s.import_rows(rows);
      if (static_cast<uint32_t>(s.len) != info.pos) throw std::runtime_error("restore: wrong number of tokens");
    }
    return;
  }

  // --- requests that run the model -----------------------------------------------
  // Find or create the session this request works on (none for OP_FORWARD).
  Session* session = nullptr;
  if (info.op == OP_PREFILL) {
    session = &sessions.create(info.session, n_layer, my_width);
  } else if (info.op == OP_DECODE) {
    session = sessions.find(info.session);
    if (!session) throw std::runtime_error("decode: this rank has no such session");
    // The root says how long it believes the session is. Agreeing on that is
    // a cheap check that every rank's cache has seen the same history.
    if (static_cast<uint32_t>(session->len) != info.pos) {
      throw std::runtime_error("decode: this rank's session length differs from the root's");
    }
  }

  const int n_new = h.rows;
  tp_forward(shard, mesh, h, &stats, session);
  // Test hook: MR_TEST_CORRUPT_RANK=R makes rank R damage its copy of h, so
  // the test suite can prove that check_identical really catches it.
  if (const char* v = std::getenv("MR_TEST_CORRUPT_RANK"); v && std::atoi(v) == mesh.rank()) h.data[0] += 1e-3f;
  mesh.check_identical(h);

  const bool argmax = (info.flags & FLAG_ARGMAX) != 0;
  const bool last_only = argmax || (info.flags & FLAG_LAST_ONLY) != 0 || info.op != OP_FORWARD;
  const auto head_t0 = std::chrono::steady_clock::now();
  const Mat local = tp_head_local(shard, h, last_only);
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
    Mat mine(1, 2);
    mine.data[0] = row[best];
    mine.data[1] = static_cast<float>(shard.vocab_begin + best);
    Mat all;
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
    // Checkpointing (Part 5): send the root a copy of the cache rows this
    // step added, so a replacement rank can be handed them later. The root's
    // own rows stay where they are.
    if (info.flags & FLAG_BACKUP) {
      const Mat rows = mesh.is_root() ? Mat() : session->export_rows(session->len - n_new, session->len);
      mesh.gather_parts(rows, out.backup_parts);
    }
    out.session_len = session->len;
    out.kv_bytes = session->bytes();
  }
}

}  // namespace mr
