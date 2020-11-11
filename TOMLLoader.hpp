#pragma once

#include <toml.hpp>

#include <string>
#include <vector>
#include <array>
#include <unordered_map>

#include "types.hpp"
#include "Sphere.hpp"
#include "HittableList.hpp"
#include "Material.hpp"
#include "Camera.hpp"
#include "Scene.hpp"

using std::vector;
using std::string;

auto
loadParams(const toml::value& scene_data)
{
    const auto raytracing_data = toml::find(scene_data, "raytracing");

    int samples = toml::find<int>(raytracing_data, "samples");
    int depth = toml::find<int>(raytracing_data, "depth");
    int height = toml::find<int>(raytracing_data, "height");
    int width = height * toml::find<num>(scene_data, "camera", "aspect_ratio");

    return std::array<int, 4>{samples, depth, width, height};
}

camera
loadCamera(const toml::value& scene_data)
{
    const auto camera_data = toml::find(scene_data, "camera");

    vector<num> lf = toml::find<vector<num>>(camera_data, "lookfrom");
    point3 lookfrom(lf[0], lf[1], lf[2]);

    vector<num> la = toml::find<vector<num>>(camera_data, "lookat");
    point3 lookat(la[0], la[1], la[2]);

    vector<num> up = toml::find<vector<num>>(camera_data, "up_vector");
    vec3 up_vector(up[0], up[1], up[2]);

    num aperture = toml::find<num>(camera_data, "aperture");

    num fov = toml::find<num>(camera_data, "fov");

    num aspect_ratio = toml::find<num>(camera_data, "aspect_ratio");

    num dist_to_focus = glm::length(lookfrom - lookat);

    return camera(lookfrom, lookat, up_vector, fov, aspect_ratio, aperture, dist_to_focus);
}

scene
loadScene(const toml::value& scene_data)
{
    scene world;

    const auto& material_data = toml::find(scene_data, "materials").as_array();

    for (const auto& mat : material_data)
    {
        std::string type = toml::find<std::string>(mat, "type");

        color c (toml::find<num>(mat, "color", 0),
                 toml::find<num>(mat, "color", 1),
                 toml::find<num>(mat, "color", 2));

        material*& ptr = world.materials[toml::find<std::string>(mat, "id")];

        switch (type[0])
        {
            case 'l':
                ptr = new lambertian(c);
                break;
            case 'm':
                ptr = new metal(c, toml::find<num>(mat, "fuzz"));
                break;
            case 'd':
                ptr = new dielectric(c, toml::find<num>(mat, "ir"));
                break;
            default:
                ptr = nullptr;
        }
    }

    const auto& object_data = toml::find(scene_data, "objects").as_array();

    for (const auto& obj : object_data)
    {
        material* mat = world.materials.at(toml::find<std::string>(obj, "material"));

        std::string geo = toml::find<std::string>(obj, "geometry");

        hittable* h = [&]()->hittable*{
            switch (geo[0])
            {
                case 's': {
                    point3 p (toml::find<num>(obj, "position", 0),
                             toml::find<num>(obj, "position", 1),
                             toml::find<num>(obj, "position", 2));
                    
                    return new sphere(p, toml::find<num>(obj, "radius"), mat);
                }
                default:
                    return nullptr;
            }
        }();
        world.objects.add(h);
    }

    return world;
}