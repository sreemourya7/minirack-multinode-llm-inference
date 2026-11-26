#include "prof.h"

namespace mr::prof {

double ms[kCount] = {0, 0, 0, 0, 0};

void reset() {
  for (double& v : ms) v = 0.0;
}

}  // namespace mr::prof
