#pragma once

#include <toml.hpp>

#include <tuple>
#include <string>
#include <unordered_map>
#include <vector>

#include "Camera.hpp"
#include "Material.hpp"
#include "Scene.hpp"
#include "Hittable.hpp"
#include "types.hpp"

using std::string;
using std::vector;

auto
loadParams (const toml::value& scene_data)
{
  const auto raytracing_data = toml::find (scene_data, "raytracing");

  int samples = toml::find<int> (raytracing_data, "samples");
  int depth = toml::find<int> (raytracing_data, "depth");
  int height = toml::find<int> (raytracing_data, "height");

  const auto camera_data = toml::find (scene_data, "camera");

  int width = height * toml::find<num> (camera_data, "aspect_ratio");
  color background(toml::find<num> (camera_data, "background" , 0),
                  toml::find<num> (camera_data, "background" , 1),
                  toml::find<num> (camera_data, "background" , 2));

  return std::tuple<int, int, int, int, color>{samples, depth, width, height, background};
}

camera
loadCamera (const toml::value& scene_data)
{
  const auto camera_data = toml::find (scene_data, "camera");

  vector<num> lf = toml::find<vector<num>> (camera_data, "lookfrom");
  point3 lookfrom (lf[0], lf[1], lf[2]);

  vector<num> la = toml::find<vector<num>> (camera_data, "lookat");
  point3 lookat (la[0], la[1], la[2]);

  vector<num> up = toml::find<vector<num>> (camera_data, "up_vector");
  vec3 up_vector (up[0], up[1], up[2]);

  num aperture = toml::find<num> (camera_data, "aperture");

  num fov = toml::find<num> (camera_data, "fov");

  num aspect_ratio = toml::find<num> (camera_data, "aspect_ratio");

  num dist_to_focus = glm::length (lookfrom - lookat);

  return camera (lookfrom, lookat, up_vector, fov, aspect_ratio, aperture,
                 dist_to_focus);
}

scene
loadScene (const toml::value& scene_data)
{
  // declare temporary unordered map along with world
  std::unordered_map<string, material*> material_map;
  scene world;

  // Getting material data
  const auto& material_data = toml::find (scene_data, "materials").as_array ();

  // Storing material data in an unordered map
  for (const auto& mat : material_data)
  {
    string type = toml::find<string> (mat, "type");

    color c (toml::find<num> (mat, "color", 0),
             toml::find<num> (mat, "color", 1),
             toml::find<num> (mat, "color", 2));

    material* ptr = nullptr;

    switch (type[0])
    {
      case 'l':
        ptr = new lambertian (c);
        break;
      case 'm':
        ptr = new metal (c, toml::find<num> (mat, "fuzz"));
        break;
      case 'd':
        ptr = new dielectric (c, toml::find<num> (mat, "ir"));
        break;
      case 'e':
        ptr = new emissive(c, toml::find<num> (mat, "intensity"));
        break;
      default:
        ptr = nullptr;
    }
    material_map[toml::find<string> (mat, "id")] = ptr;
  }

  // Converting unordered map into scene array
  world.material_count = material_map.size();
  world.materials = new material*[world.material_count];
  int i = 0;
  for (const auto& [key, value] : material_map)
  {
    world.materials[i] = value;
    ++i;
  }
  const auto& object_data = toml::find (scene_data, "objects").as_array ();

  world.object_count = object_data.size();
  world.objects = new hittable*[world.object_count];

  i = 0;
  for (const auto& obj : object_data)
  {
    // Referencing material_map to find correct index.
    std::string key = toml::find<std::string> (obj, "material");
    int mat_index =
      std::distance (std::begin (material_map), material_map.find (key));

    std::string geo = toml::find<std::string> (obj, "geometry");

    hittable* h = [&] () -> hittable* {
      switch (geo[0])
      {
        case 's': // Sphere
        {
          point3 p (toml::find<num> (obj, "position", 0),
                    toml::find<num> (obj, "position", 1),
                    toml::find<num> (obj, "position", 2));
          return new sphere (p, toml::find<num> (obj, "radius"), mat_index);
        }
        case 'p': // Plane
        {
          point3 p (toml::find<num> (obj, "position", 0),
                    toml::find<num> (obj, "position", 1),
                    toml::find<num> (obj, "position", 2));
          vec3 n (toml::find<num> (obj, "normal", 0),
                  toml::find<num> (obj, "normal", 1),
                  toml::find<num> (obj, "normal", 2));
          return new plane (p, n, mat_index);
        }
        case 't': // Triangle
        {
          point3 p1 (toml::find<num> (obj, "p1", 0),
                    toml::find<num> (obj, "p1", 1),
                    toml::find<num> (obj, "p1", 2));
          point3 p2 (toml::find<num> (obj, "p2", 0),
                    toml::find<num> (obj, "p2", 1),
                    toml::find<num> (obj, "p2", 2));
          point3 p3 (toml::find<num> (obj, "p3", 0),
                    toml::find<num> (obj, "p3", 1),
                    toml::find<num> (obj, "p3", 2));
          return new triangle (p1, p2, p3, mat_index);
        }
        default:
          return nullptr;
      }
    }();
    world.objects[i] = h;
    ++i;
  }

  return world;
}