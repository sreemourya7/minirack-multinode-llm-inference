#include "weights.h"

#include <cstring>
#include <stdexcept>

namespace mr {

namespace {
void read_exact(FILE* f, void* dst, size_t n) {
  if (std::fread(dst, 1, n, f) != n) throw std::runtime_error("weight file: unexpected end of file");
}
uint32_t read_u32(FILE* f) {
  uint32_t v;
  read_exact(f, &v, sizeof v);
  return v;
}
uint64_t read_u64(FILE* f) {
  uint64_t v;
  read_exact(f, &v, sizeof v);
  return v;
}
}  // namespace

WeightFile::WeightFile(const std::string& path) {
  f_ = std::fopen(path.c_str(), "rb");
  if (!f_) throw std::runtime_error("cannot open weight file: " + path);

  char magic[4];
  read_exact(f_, magic, 4);
  if (std::memcmp(magic, "MRKW", 4) != 0) throw std::runtime_error("not a .mrkw file: " + path);
  const uint32_t version = read_u32(f_);
  if (version != 1) throw std::runtime_error("unsupported .mrkw version");

  config_.n_layer = read_u32(f_);
  config_.n_head = read_u32(f_);
  config_.n_embd = read_u32(f_);
  config_.vocab_size = read_u32(f_);
  config_.n_ctx = read_u32(f_);

  const uint32_t n = read_u32(f_);
  for (uint32_t i = 0; i < n; ++i) {
    const uint32_t name_len = read_u32(f_);
    if (name_len > 4096) throw std::runtime_error("weight file: bad name length");
    std::string name(name_len, '\0');
    read_exact(f_, name.data(), name_len);
    Entry e;
    e.rows = read_u32(f_);
    e.cols = read_u32(f_);
    e.offset = read_u64(f_);
    index_.emplace(std::move(name), e);
  }
}

WeightFile::~WeightFile() {
  if (f_) std::fclose(f_);
}

Mat WeightFile::get(const std::string& name) {
  auto it = index_.find(name);
  if (it == index_.end()) throw std::runtime_error("tensor not in weight file: " + name);
  const Entry& e = it->second;
  Mat m(static_cast<int>(e.rows), static_cast<int>(e.cols));
  if (fseeko(f_, static_cast<off_t>(e.offset), SEEK_SET) != 0) throw std::runtime_error("weight file: seek failed");
  const size_t nbytes = m.size() * sizeof(float);
  read_exact(f_, m.data.data(), nbytes);
  bytes_loaded_ += nbytes;
  return m;
}

}  // namespace mr
