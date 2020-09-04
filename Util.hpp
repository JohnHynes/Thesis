#pragma once

#include <iostream>
#include <random>

#include "Ray.hpp"

// Constants
constexpr float infinity = std::numeric_limits<float>::infinity();
constexpr float pi = 3.14159265358979323846264338327950;

// Color Functions
inline void printAsColor(const glm::vec3 &v)
{
    std::cout << int(v.r * 255) << ' ' << int(v.g * 255) << ' ' << int(v.b * 255) << std::endl;
}

inline glm::vec3 blend_color(const ray &r)
{
    float t = 0.5f * (glm::normalize(r.dir).y + 1.0f);
    return (1.0f - t) * glm::vec3(1.0f, 1.0f, 1.0f) + t * glm::vec3(0.5, 0.7, 1.0);
}

// Utility Functions
inline float degrees_to_radians(float degrees) {
    return degrees * pi / 180.0f;
}

inline float random_float() {
    static std::uniform_real_distribution<double> distribution(0.0, 1.0);
    static std::mt19937 generator;
    return distribution(generator);
}

