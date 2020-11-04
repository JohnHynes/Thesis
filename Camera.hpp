#pragma once

#include <glm/vec3.hpp>

#include "types.hpp"
#include "constants.hpp"
#include "Util.hpp"

class camera {
private:
    point3 origin;
    point3 lower_left_corner;
    vec3 horizontal;
    vec3 vertical;
    vec3 u, v, w;
    num lens_radius;

public:
    camera(const point3& lookfrom, const point3& lookat, const vec3& vup, const num& vfov,
        const num& aspect_ratio, const num& aperture, const num& focus_dist)
    {
        set(lookfrom, lookat, vup, vfov, aspect_ratio, aperture, focus_dist);
    }

    camera (camera const &) = default;
    camera (camera &&) = default;
    camera &operator= (camera const &) = default;
    camera &operator= (camera &&) = default;

    void
    set(const point3& lookfrom, const point3& lookat, const vec3& vup, const num& vfov,
        const num& aspect_ratio, const num& aperture, const num& focus_dist)
    {
        num theta = degrees_to_radians(vfov);
        num h = tan(theta / CONST(2));
        num viewport_height = CONST(2) * h;
        num viewport_width = aspect_ratio * viewport_height;

        w = glm::normalize(lookfrom - lookat);
        u = glm::normalize(glm::cross(vup, w));
        v = glm::cross(w, u);

        origin = lookfrom;
        horizontal = focus_dist * viewport_width * u;
        vertical = focus_dist * viewport_height * v;
        lower_left_corner = origin - horizontal / CONST(2) - vertical / CONST(2) - focus_dist*w;

        lens_radius = aperture / CONST(2);
    }

    ray get_ray(num s, num t) const {
        vec3 rd = lens_radius * random_in_unit_disk();
        vec3 offset = u * rd.x + v * rd.y;
        return ray(
            origin + offset,
            lower_left_corner + s*horizontal + t*vertical - origin - offset
        );
    }
private:  
    vec3 random_in_unit_disk() const {
        while (true) {
            vec3 p = vec3(rng.random_unit(), rng.random_unit(), CONST(0));
            if (glm::dot(p,p) >= 1) continue;
            return p;
        }
    }
};