cmake_minimum_required(VERSION 3.16)
project(minirack CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Part 1 is the plain scalar baseline: -O2, no -march=native, no -ffast-math.
# Hand-written SIMD arrives in Part 5 and is measured against this.
set(CMAKE_CXX_FLAGS_RELEASE "-O2")
add_compile_options(-Wall -Wextra -Wpedantic)
add_compile_definitions(_FILE_OFFSET_BITS=64 _DEFAULT_SOURCE)

add_library(mr_core STATIC
  src/ops.cpp
  src/weights.cpp
  src/model.cpp
  src/net.cpp
)
target_include_directories(mr_core PUBLIC src)

add_executable(mr_worker src/worker.cpp)
target_link_libraries(mr_worker PRIVATE mr_core)
