#pragma once

#include <glm/vec3.hpp>
#include <glm/vec4.hpp>

class Ray {
    public:
    // Constructors
    Ray() = default;
    constexpr Ray(Ray const&) = default;
    constexpr Ray(Ray &&) = default;
    constexpr Ray& operator=(Ray const&) = default;
    constexpr Ray& operator=(Ray &&) = default;

    Ray (const glm::vec3& neworigin, const glm::vec3& newdir, const glm::vec3& newcolor)
    : origin(neworigin), dir(newdir), color(newcolor) {}

    // Members
    glm::vec3 origin;
    glm::vec3 dir;
    glm::vec3 color;
};