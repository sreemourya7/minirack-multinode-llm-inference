cmake_minimum_required(VERSION 3.16)
project(minirack CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Everything is built with plain -O2: no -march=native, no -ffast-math. The
# scalar kernels are the baseline; the AVX2 kernels (below) are measured against them.
set(CMAKE_CXX_FLAGS_RELEASE "-O2")
add_compile_options(-Wall -Wextra -Wpedantic)
add_compile_definitions(_FILE_OFFSET_BITS=64 _DEFAULT_SOURCE)

add_library(mr_core STATIC
  src/ops.cpp
  src/weights.cpp
  src/model.cpp
  src/net.cpp
  src/mesh.cpp
  src/tp_model.cpp
  src/moe.cpp
  src/kv_cache.cpp
  src/exec.cpp
  src/root.cpp
  src/kernels.cpp
  src/prof.cpp
)

# Part 6: hand-written AVX2 kernels. Only this one file is compiled with
# -mavx2 -mfma, and it is only called after a run-time check of the CPU, so
# the binary still runs on an x86-64 machine without AVX2.
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64")
  target_sources(mr_core PRIVATE src/kernels_avx2.cpp)
  set_source_files_properties(src/kernels_avx2.cpp PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma")
  target_compile_definitions(mr_core PUBLIC MR_HAVE_AVX2)
endif()
target_include_directories(mr_core PUBLIC src)

add_executable(mr_worker src/worker.cpp)
target_link_libraries(mr_worker PRIVATE mr_core)

# Part 2: one rank of the tensor-parallel rack.
add_executable(mr_rank src/rank.cpp)
target_link_libraries(mr_rank PRIVATE mr_core)

# Part 6: kernel unit tests and microbenchmarks.
add_executable(mr_kernel_test src/kernel_test.cpp)
target_link_libraries(mr_kernel_test PRIVATE mr_core)
add_executable(mr_bench src/bench.cpp)
target_link_libraries(mr_bench PRIVATE mr_core)
