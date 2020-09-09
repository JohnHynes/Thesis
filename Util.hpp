#pragma once

#include <iostream>
#include <random>

#include "types.hpp"
#include "constants.hpp"

#include "Ray.hpp"
#include "Vec3.hpp"
#include "Random.hpp"

static random_gen rng;

// Utility Functions

inline precision degrees_to_radians(precision degrees)
{
    return degrees * pi / 180.0;
}

inline precision clamp(precision x, precision min, precision max)
{
    if (x < min)
        return min;
    if (x > max)
        return max;
    return x;
}

glm::vec3 random_unit_vector()
{
    precision a = rng.random_angle();
    precision z = rng.random_unit();
    precision r = sqrt(1 - z * z);
    return glm::vec3(r * cos(a), r * sin(a), z);
}

glm::vec3 random_in_hemisphere(const glm::vec3 &normal)
{
    glm::vec3 in_unit_sphere = random_unit_vector();
    if (glm::dot(in_unit_sphere, normal) > 0.0)
        return in_unit_sphere;
    else
        return -in_unit_sphere;
}

// Color Functions

inline void write_color(std::ostream &out, const glm::vec3 &color, int samples_per_pixel)
{
    precision scale = 1.0 / samples_per_pixel;

    precision r = sqrt(color.r * scale);
    precision g = sqrt(color.g * scale);
    precision b = sqrt(color.b * scale);

    out << static_cast<int>(256 * clamp(r, 0, 0.999999)) << ' '
        << static_cast<int>(256 * clamp(g, 0, 0.999999)) << ' '
        << static_cast<int>(256 * clamp(b, 0, 0.999999)) << '\n';
}
