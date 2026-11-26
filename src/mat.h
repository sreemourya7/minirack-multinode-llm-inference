// mat.h - the one data container used everywhere in Part 1.
//
// A Mat is a dense, row-major, float32 matrix. "Row-major" means row 0 is
// stored first, then row 1, and so on, in one flat array. Every activation in
// a transformer forward pass fits this shape: [tokens, features].
#pragma once

#include <cstddef>
#include <vector>

namespace mr {

struct Mat {
  int rows = 0;
  int cols = 0;
  std::vector<float> data;

  Mat() = default;
  Mat(int r, int c) : rows(r), cols(c), data(static_cast<size_t>(r) * c, 0.0f) {}

  float* row(int r) { return data.data() + static_cast<size_t>(r) * cols; }
  const float* row(int r) const { return data.data() + static_cast<size_t>(r) * cols; }
  size_t size() const { return data.size(); }
};

}  // namespace mr
