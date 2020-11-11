#pragma once

#include <vector>

#include "types.hpp"
#include "Hittable.hpp"

class hittable_list : public hittable
{
public:
    std::vector<hittable*> objects;

public:
    // Constructors
    hittable_list(){};
    hittable_list(hittable_list const &) = default;
    hittable_list(hittable_list &&) = default;
    hittable_list &operator=(hittable_list const &) = default;
    hittable_list &operator=(hittable_list &&) = default;

    hittable_list(hittable* object) { add(object); };

    // Destructor
    ~hittable_list()
    {
        for (auto& p :objects)
        {
            delete p;
        }
    }

    // Mutators
    void clear() { objects.clear(); }
    void add(hittable* object) { objects.push_back(object); }

    // Member Functions
    bool hit(const ray &r, num tmin, num tmax, hit_record &hitrec) const
    {
        hit_record temp_hitrec;
        bool has_hit = false;
        num closest_seen = tmax;

        for (const auto &object : objects)
        {
            if (object->hit(r, tmin, closest_seen, temp_hitrec))
            {
                has_hit = true;
                closest_seen = temp_hitrec.t;
                hitrec = temp_hitrec;
            }
        }

        return has_hit;
    }
};