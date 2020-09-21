#pragma once

#include <glm/vec3.hpp>
#include <glm/vec4.hpp>

#include "types.hpp"
#include "Vec3.hpp"

class ray
{
public:
    point3 origin;
    glm::vec3 dir;

public:
    // Constructors
    ray() = default;
    constexpr ray(ray const &) = default;
    constexpr ray(ray &&) = default;
    constexpr ray &operator=(ray const &) = default;
    constexpr ray &operator=(ray &&) = default;

    ray(const point3 &neworigin, const glm::vec3 &newdir)
        : origin(neworigin), dir(newdir) {}

    // Member Functions
    point3 at(precision t) const
    {
        return origin + t * dir;
    }
};