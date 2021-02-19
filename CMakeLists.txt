project(RayTracing LANGUAGES CXX CUDA)

cmake_minimum_required(VERSION 3.18)

set(CMAKE_BUILD_TYPE Release)

enable_language(CUDA)
include(CheckLanguage)
check_language(CUDA)

find_package(OpenMP REQUIRED)

add_subdirectory(glm)
add_subdirectory(toml11)

add_executable(main main.cpp)
add_executable(main_cuda main.cu)

target_compile_options(main
  PRIVATE
  "-march=native")

set_target_properties(main_cuda PROPERTIES CUDA_ARCHITECTURES 61)

target_compile_options(main_cuda
  PRIVATE
  "-DUSE_GPU=1"
  "--expt-relaxed-constexpr"
  "--resource-usage"
  "--restrict"
  "-Xptxas=\"-v\"")

target_compile_features(main
  PUBLIC
  cxx_std_17)

target_compile_features(main_cuda
  PUBLIC
  cuda_std_17)

target_link_libraries(main
  OpenMP::OpenMP_CXX
  glm::glm
  toml11::toml11)

target_link_libraries(main_cuda
  glm::glm
  toml11::toml11)
  