#pragma once

#include "glm/glm.hpp"
#include <cmath>
#include <iostream>
#include <random>

#include "constants.hpp"
#include "types.hpp"

#include "Random.hpp"
#include "Ray.hpp"

thread_local random_gen rng;

// Utility Functions

inline precision
degrees_to_radians (precision degrees)
{
  return degrees * pi / 180.0;
}

vec3
random_unit_vector ()
{
  precision a = rng.random_angle ();
  precision z = rng.random_unit ();
  precision r = sqrt (1 - z * z);
  return vec3 (r * cos (a), r * sin (a), z);
}

vec3
random_in_hemisphere (const vec3 &normal)
{
  vec3 in_unit_sphere = random_unit_vector ();
  if (glm::dot (in_unit_sphere, normal) > 0.0)
    return in_unit_sphere;
  else
    return -in_unit_sphere;
}

// Color Functions

inline void
write_color (std::ostream &out, const color &c, int samples_per_pixel)
{
  precision scale = 1.0 / samples_per_pixel;

  precision r = sqrt (c.r * scale);
  precision g = sqrt (c.g * scale);
  precision b = sqrt (c.b * scale);

  const precision lower = 0;
  const precision upper = 0.99999;
  out << static_cast<int> (256 * glm::clamp (r, lower, upper)) << ' '
      << static_cast<int> (256 * glm::clamp (g, lower, upper)) << ' '
      << static_cast<int> (256 * glm::clamp (b, lower, upper)) << '\n';
}
