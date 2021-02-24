#ifndef GPU_RAY_TRACING_RANDOM_HPP_
#define GPU_RAY_TRACING_RANDOM_HPP_

#include "Preprocessor.hpp"
#include "constants.hpp"
#include "types.hpp"

#include <glm/geometric.hpp>

using RandomState = void*;

#include <random>
using RandomStateCPU = std::mt19937;

#ifdef USE_GPU
#include <curand.h>
#include <curand_kernel.h>
using RandomStateGPU = curandState;
#endif

__host__ __device__
static
inline
num
get_next_rand(RandomState* s) {
  #if defined(__CUDA_ARCH__) && defined(__CUDACC__)
    return curand_uniform (reinterpret_cast<RandomStateGPU*>(s));
  #else
    thread_local std::uniform_real_distribution<num> dist{CONST(0.0), CONST(1.0)};
    return dist (*reinterpret_cast<RandomStateCPU*>(s));
  #endif
}

__host__ __device__
inline
num
random_positive_unit (RandomState* s)
{
  return get_next_rand(s);
}

__host__ __device__
inline
num
random_unit (RandomState* s)
{
  return get_next_rand(s) * 2.0 - 1.0;
}

__host__ __device__
inline
num
random_angle (RandomState* s)
{
  return get_next_rand(s) * CONST(2.0) * pi;
}

__host__ __device__
inline
int
random_int (RandomState* s, int low, int high)
{
  return static_cast<int> (random_positive_unit (s) * (high - low) + low);
}

__host__ __device__
inline
vec3
random_unit_vector (RandomState* s)
{
  num a = random_angle (s);
  num z = random_unit (s);
  num r = sqrt (num (1) - z * z);
  return vec3 (r * cos (a), r * sin (a), z);
}

__host__ __device__
inline
vec3
random_in_unit_sphere (RandomState* s)
{
  while (true)
  {
    vec3 p {random_unit (s), random_unit (s), random_unit (s)};
    if (glm::dot (p, p) >= 1)
      continue;
    return p;
  }
}

__host__ __device__
inline
vec3
random_in_hemisphere (RandomState* s, const vec3 &normal)
{
  vec3 in_unit_sphere = random_unit_vector (s);
  if (glm::dot (in_unit_sphere, normal) > 0.0)
    return in_unit_sphere;
  else
    return -in_unit_sphere;
}

#endif