// weights.h - reader for the minirack weight file (.mrkw).
//
// File layout (all integers little-endian):
//
//   "MRKW"            4 bytes magic
//   u32 version       currently 1
//   u32 n_layer, n_head, n_embd, vocab_size, n_ctx
//   u32 n_tensors
//   n_tensors index entries:
//       u32 name_len, name bytes, u32 rows, u32 cols, u64 offset
//   raw float32 data
//
// The index sits at the front so a worker can look up a tensor by name and
// read ONLY the bytes it needs. That is what lets each worker hold just its
// own shard of the model instead of the whole thing.
#pragma once

#include <cstdint>
#include <cstdio>
#include <string>
#include <unordered_map>

#include "mat.h"

namespace mr {

struct ModelConfig {
  uint32_t n_layer = 0;
  uint32_t n_head = 0;
  uint32_t n_embd = 0;
  uint32_t vocab_size = 0;
  uint32_t n_ctx = 0;
};

class WeightFile {
 public:
  explicit WeightFile(const std::string& path);
  ~WeightFile();
  WeightFile(const WeightFile&) = delete;
  WeightFile& operator=(const WeightFile&) = delete;

  const ModelConfig& config() const { return config_; }

  // Reads one tensor from disk. Throws if the name is not in the file.
  Mat get(const std::string& name);

  // Total bytes read through get() so far (to show how big a shard is).
  uint64_t bytes_loaded() const { return bytes_loaded_; }

 private:
  struct Entry {
    uint32_t rows, cols;
    uint64_t offset;
  };
  FILE* f_ = nullptr;
  ModelConfig config_;
  std::unordered_map<std::string, Entry> index_;
  uint64_t bytes_loaded_ = 0;
};

}  // namespace mr
