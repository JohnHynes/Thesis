#pragma once

#include <glm/vec3.hpp>
#include <glm/vec4.hpp>

#include "types.hpp"

class ray
{
public:
    point3 origin;
    vec3 dir;

public:
    // Constructors
    ray() = default;
    constexpr ray(ray const &) = default;
    constexpr ray(ray &&) = default;
    constexpr ray &operator=(ray const &) = default;
    constexpr ray &operator=(ray &&) = default;

    ray(const point3 &neworigin, const vec3 &newdir)
        : origin(neworigin), dir(newdir) {}

    // Member Functions
    point3 at(num t) const
    {
        return origin + t * dir;
    }
};