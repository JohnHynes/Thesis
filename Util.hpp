#pragma once

#include <iostream>

#include "Ray.hpp"

inline
float dot (const Ray& a, const Ray& b){
    return a.dir.x * b.dir.x + a.dir.y * b.dir.y + a.dir.z * b.dir.z;
}

inline
glm::vec3 cross (const Ray& a, const Ray& b){
    return glm::vec3(a.dir.y * b.dir.z - a.dir.z * b.dir.y,
                     a.dir.z * b.dir.x - a.dir.x * b.dir.z,
                     a.dir.x * b.dir.y - a.dir.y * b.dir.x);
}

inline
glm::vec3 cross (const glm::vec3& a, const glm::vec3& b){
    return glm::vec3(a.y * b.z - a.z * b.y,
                     a.z * b.x - a.x * b.z,
                     a.x * b.y - a.y * b.x);
}

inline
void printAsColor(const glm::vec3& v){
    std::cout << int(v.r * 255) << ' ' << int(v.g * 255) << ' ' << int(v.b * 255) << std::endl;
}

// (1 - t) * startValue + t * endValue
inline
glm::vec3 blend_color(const Ray& v) {
    float t = 0.5f * (glm::normalize(v.dir).y + 1.0f);
    return (1.0f - t) * glm::vec3(1.0f, 1.0f, 1.0f) + t * glm::vec3(0.5, 0.7,1.0);
}