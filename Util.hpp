#pragma once

#include <iostream>
#include <random>

#include "Ray.hpp"
#include "Vec3.hpp"

// Constants

constexpr double infinity = std::numeric_limits<double>::infinity();
constexpr double pi = 3.14159265358979323846264338327950;

// Utility Functions

inline double degrees_to_radians(double degrees)
{
    return degrees * pi / 180.0;
}

inline double random_double()
{
    static std::uniform_real_distribution<double> distribution(0.0, 1.0);
    static std::mt19937 generator;
    return distribution(generator);
}

inline double random_double(double min, double max)
{
    static std::uniform_real_distribution<double> distribution(min, max);
    static std::mt19937 generator;
    return distribution(generator);
}

inline double clamp(double x, double min, double max)
{
    if (x < min)
        return min;
    if (x > max)
        return max;
    return x;
}

inline static glm::vec3 random_vector()
{
    return glm::vec3(random_double(), random_double(), random_double());
}

inline static glm::vec3 random_vector(double min, double max)
{
    return glm::vec3(random_double(min, max), random_double(min, max), random_double(min, max));
}

glm::vec3 random_unit_vector()
{
    double a = random_double(0, 2 * pi);
    double z = random_double(-1, 1);
    double r = sqrt(1 - z * z);
    return glm::vec3(r * cos(a), r * sin(a), z);
}

glm::vec3 random_in_unit_sphere()
{
    while (true)
    {
        auto p = random_vector(-1, 1);
        if (glm::dot(p, p) >= 1)
            continue;
        return p;
    }
}

glm::vec3 random_in_hemisphere(const glm::vec3 &normal)
{
    glm::vec3 in_unit_sphere = random_in_unit_sphere();
    if (glm::dot(in_unit_sphere, normal) > 0.0)
        return in_unit_sphere;
    else
        return -in_unit_sphere;
}

// Color Functions

inline void write_color(std::ostream &out, const glm::vec3 &color, int samples_per_pixel)
{
    double scale = 1.0f / samples_per_pixel;

    double r = sqrt(color.r * scale);
    double g = sqrt(color.g * scale);
    double b = sqrt(color.b * scale);

    out << static_cast<int>(256 * clamp(r, 0, 0.999999)) << ' '
        << static_cast<int>(256 * clamp(g, 0, 0.999999)) << ' '
        << static_cast<int>(256 * clamp(b, 0, 0.999999)) << '\n';
}

inline glm::vec3 blend_color(const ray &r)
{
    double t = 0.5 * (glm::normalize(r.dir).y + 1.0);
    return (1.0 - t) * glm::vec3(1.0f, 1.0f, 1.0f) + t * glm::vec3(0.5, 0.7, 1.0);
}
