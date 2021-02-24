#ifndef GPU_RAY_TRACING_UTIL_HPP_
#define GPU_RAY_TRACING_UTIL_HPP_

#include <cmath>
#include <iostream>
#include <algorithm>

#include "constants.hpp"
#include "types.hpp"

// Utility Functions

inline num
degrees_to_radians (num degrees)
{
  return degrees * pi / 180.0;
}

// Color Functions
inline void
write_color (std::ostream &out, const color &c, int samples_per_pixel)
{
  num scale = num(1) / samples_per_pixel;

  num r = sqrt (c.r * scale);
  num g = sqrt (c.g * scale);
  num b = sqrt (c.b * scale);

  const num lower = 0;
  const num upper = 0.99999;
  out << static_cast<int> (256 * std::clamp (r, lower, upper)) << ' '
      << static_cast<int> (256 * std::clamp (g, lower, upper)) << ' '
      << static_cast<int> (256 * std::clamp (b, lower, upper)) << '\n';
}

#endif
