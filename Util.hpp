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
  return degrees * (pi / CONST(180.0));
}

// Color Functions
inline void
write_color (std::ostream &out, const color &c, int samples_per_pixel)
{
  double scale = 1.0/ static_cast<double>(samples_per_pixel);

  double r = sqrt (c.r * scale);
  double g = sqrt (c.g * scale);
  double b = sqrt (c.b * scale);
  const double lower = 0;
  const double upper = 0.99999;
  out << static_cast<int> (256 * std::clamp (r, lower, upper)) << ' '
      << static_cast<int> (256 * std::clamp (g, lower, upper)) << ' '
      << static_cast<int> (256 * std::clamp (b, lower, upper)) << '\n';
}

#endif
