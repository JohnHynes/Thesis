#ifndef GPU_RAY_TRACER_CONSTANTS_HPP_
#define GPU_RAY_TRACER_CONSTANTS_HPP_

#include <limits>
#include <glm/vec3.hpp>

#include "types.hpp"

// Mathematical constants
constexpr num infinity = std::numeric_limits<num>::infinity();
constexpr num pi {3.14159265358979323846264338327950};

constexpr vec3 origin (0.0, 0.0, 0.0);

#endif