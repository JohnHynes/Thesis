#pragma once

#include <glm/vec3.hpp>
#include <glm/vec4.hpp>

#include "types.hpp"

class ray
{
public:
    point3 orig;
    vec3 dir;

public:
    // Constructors
    HOST_DEVICE
    ray() : orig(), dir() {}
    
    constexpr ray(ray const &) = default;
    constexpr ray(ray &&) = default;
    constexpr ray &operator=(ray const &) = default;
    constexpr ray &operator=(ray &&) = default;

    HOST_DEVICE 
    ray(const point3 &neworigin, const vec3 &newdir)
        : orig(neworigin), dir(newdir) {}

    HOST_DEVICE 
    point3 origin() const  { return orig; }
    
    HOST_DEVICE 
    vec3 direction() const { return dir; }

    // Member Functions
    HOST_DEVICE
    point3 at(num t) const
    {
        return orig + t * dir;
    }
};