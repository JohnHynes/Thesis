#pragma once

#include <iostream>
#include <random>

#include "Ray.hpp"

// Constants
constexpr float infinity = std::numeric_limits<float>::infinity();
constexpr float pi = 3.14159265358979323846264338327950;

// Utility Functions
inline float degrees_to_radians (float degrees) {
    return degrees * pi / 180.0f;
}

inline float random_float () {
    static std::uniform_real_distribution<float> distribution(0.0, 1.0);
    static std::mt19937 generator;
    return distribution(generator);
}

inline double clamp (float x, float min, float max)
{
    if (x < min) return min;
    if (x > max) return max;
    return x;
}

// Color Functions
inline void write_color (std::ostream& out, const glm::vec3 &v, int samples_per_pixel)
{
    float scale = 1.0f / samples_per_pixel;

    out << static_cast<int>(256 * clamp(v.r * scale, 0, 0.99999)) << ' '
        << static_cast<int>(256 * clamp(v.g * scale, 0, 0.99999)) << ' '
        << static_cast<int>(256 * clamp(v.b * scale, 0, 0.99999)) << '\n';
}

inline glm::vec3 blend_color(const ray &r)
{
    float t = 0.5f * (glm::normalize(r.dir).y + 1.0f);
    return (1.0f - t) * glm::vec3(1.0f, 1.0f, 1.0f) + t * glm::vec3(0.5, 0.7, 1.0);
}
