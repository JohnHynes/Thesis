#pragma once

#include <glm/vec3.hpp>
#include <glm/vec4.hpp>

class ray
{
public:
    glm::vec3 origin;
    glm::vec3 dir;

public:
    // Constructors
    ray() = default;
    constexpr ray(ray const &) = default;
    constexpr ray(ray &&) = default;
    constexpr ray &operator=(ray const &) = default;
    constexpr ray &operator=(ray &&) = default;

    ray(const glm::vec3 &neworigin, const glm::vec3 &newdir)
        : origin(neworigin), dir(newdir) {}

    // Member Functions
    glm::vec3 at(float t) const
    {
        return origin + t * dir;
    }
};