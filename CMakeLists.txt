project(RayTracing LANGUAGES CXX)

cmake_minimum_required(VERSION 3.18)

set(CMAKE_BUILD_TYPE Release)

set(glm_DIR glm/cmake/glm)

find_package(glm REQUIRED)
find_package(OpenMP REQUIRED)

add_subdirectory(toml11)

add_executable(main main.cpp)

target_compile_options(main
  PRIVATE
  "-march=native"
  "-std=c++17")

target_link_libraries(main
  OpenMP::OpenMP_CXX
  glm::glm
  toml11::toml11)
