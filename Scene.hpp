#pragma once

#include <unordered_map>
#include <string>


#include "types.hpp"
#include "HittableList.hpp"

class scene
{
public:
    hittable_list objects;
    std::unordered_map<std::string, material*> materials;
};