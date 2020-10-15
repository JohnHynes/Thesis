#pragma once

#include <glm/vec3.hpp>

#include "types.hpp"
#include "constants.hpp"
#include "Util.hpp"

class camera
{
private:
  point3 cam_origin;
  point3 lower_left_corner;
  vec3 horizontal;
  vec3 vertical;
  vec3 u, v, w;
  precision lens_radius;

public:
  // Constructors
  camera (const point3& lookfrom, const point3& lookat, const vec3 up, precision fov, precision aspect_ratio, precision aperture, precision focus_dist) {
    precision theta = degrees_to_radians(fov);
    precision h = tan(theta / 2);
    precision viewport_height = 2.0 * h;
    precision viewport_width = aspect_ratio * viewport_height;

    w = glm::normalize (lookfrom - lookat);
    u = glm::normalize (glm::cross(up,w));
    v = glm::cross (w, u);

    cam_origin = lookfrom;
    lens_radius = aperture / 2;
    horizontal = u * viewport_width * focus_dist;
    vertical = v * viewport_height * focus_dist;
    lower_left_corner = cam_origin - horizontal / CONST(2) - vertical / CONST(2) - w * focus_dist;
  }

  camera (camera const &) = default;
  camera (camera &&) = default;
  camera &operator= (camera const &) = default;
  camera &operator= (camera &&) = default;

  // Member Functions
  ray get_ray(precision a, precision b)
  {
    vec3 rv = lens_radius * random_in_unit_disk();
    vec3 offset = u * rv.x + v * rv.y;
    return ray(cam_origin + offset, lower_left_corner + a * horizontal + b * vertical - cam_origin - offset);
  }
  
  private:
  
  vec3 random_in_unit_disk() {
    while (true) {
      vec3 p = vec3(rng.random_unit(), rng.random_unit(), 0);
      if (glm::dot(p,p) < 1) {
        return p;
      }
    }
  }
};