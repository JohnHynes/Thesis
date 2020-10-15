project(RayTracing LANGUAGES CXX)

cmake_minimum_required(VERSION 3.18)

set(CMAKE_BUILD_TYPE Release)

set(glm_DIR glm/cmake/glm)
find_package(glm REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(main main.cpp)

target_compile_options(main
  PRIVATE
  "-march=native")

target_link_libraries(main
  OpenMP::OpenMP_CXX
  glm::glm)
