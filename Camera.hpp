#pragma once

#include <glm/vec3.hpp>

#include "types.hpp"
#include "Util.hpp"

class camera
{
private:
  glm::vec3 origin;
  glm::vec3 lower_left_corner;
  glm::vec3 horizontal;
  glm::vec3 vertical;

public:
  // Constructors
  camera () {
      precision aspect_ratio = 16.0/ 10.0;
      precision viewport_height = 2.0;
      precision viewport_width = aspect_ratio * viewport_height;
      precision focal_length = 1.0;

      origin = glm::vec3(0.0, 0.0, 0.0);
      horizontal = glm::vec3(viewport_width, 0.0, 0.0);
      vertical = glm::vec3(0.0, viewport_height, 0.0);
      lower_left_corner = origin - horizontal / 2.0 - vertical / 2.0 - glm::vec3(0.0, 0.0, focal_length);
  }

  camera (camera const &) = default;
  constexpr camera (camera &&) = default;
  constexpr camera &operator= (camera const &) = default;
  constexpr camera &operator= (camera &&) = default;

  // Member Functions
  ray get_ray(precision u, precision v) const
  {
      return ray(origin, lower_left_corner + u * horizontal + v * vertical - origin);
  }
};