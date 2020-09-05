#pragma once

#include "Util.hpp"
#include <glm/vec3.hpp>

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
      float aspect_ratio = 16.0/ 10.0;
      float viewport_height = 2.0;
      float viewport_width = aspect_ratio * viewport_height;
      float focal_length = 1.0;

      origin = glm::vec3(0.0f, 0.0f, 0.0f);
      horizontal = glm::vec3(viewport_width, 0.0f, 0.0f);
      vertical = glm::vec3(0.0f, viewport_height, 0.0f);
      lower_left_corner = origin - horizontal / 2.0f - vertical / 2.0f - glm::vec3(0.0f, 0.0f, focal_length);
  }

  camera (camera const &) = default;
  constexpr camera (camera &&) = default;
  constexpr camera &operator= (camera const &) = default;
  constexpr camera &operator= (camera &&) = default;

  // Member Functions
  ray get_ray(float u, float v) const
  {
      return ray(origin, lower_left_corner + u * horizontal + v * vertical - origin);
  }
};