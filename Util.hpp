#pragma once

#include "glm/glm.hpp"
#include <cmath>
#include <iostream>
#include <random>

#include "constants.hpp"
#include "types.hpp"

#include "Random.hpp"
#include "Ray.hpp"

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
  out << static_cast<int> (256 * glm::clamp (r, lower, upper)) << ' '
      << static_cast<int> (256 * glm::clamp (g, lower, upper)) << ' '
      << static_cast<int> (256 * glm::clamp (b, lower, upper)) << '\n';
}
