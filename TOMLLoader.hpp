#ifndef GPU_RAY_TRACING_TOMLLOADER_HPP_
#define GPU_RAY_TRACING_TOMLLOADER_HPP_

#include <toml.hpp>

#include <tuple>

#include "Camera.hpp"
#include "Scene.hpp"
#include "Random.hpp"

auto
loadParams (const toml::value& scene_data) -> std::tuple<int, int, int, int, color>;

auto
loadCamera (const toml::value& scene_data) -> camera;

auto
loadScene (const toml::value& scene_data) -> scene;

// TODO: move the below to a source file

#include <string>
#include <unordered_map>
#include <vector>

#include "Hittable.hpp"
#include "Material.hpp"

using std::string;
using std::vector;

inline auto
loadParams (const toml::value& scene_data) -> std::tuple<int, int, int, int, color>
{
  const auto raytracing_data = toml::find (scene_data, "raytracing");

  int samples = toml::find<int> (raytracing_data, "samples");
  int depth = toml::find<int> (raytracing_data, "depth");
  int height = toml::find<int> (raytracing_data, "height");

  const auto camera_data = toml::find (scene_data, "camera");

  int width = static_cast<int> (height * toml::find<num> (camera_data, "aspect_ratio"));

  color background (toml::find<num> (camera_data, "background_color", 0),
                    toml::find<num> (camera_data, "background_color", 1),
                    toml::find<num> (camera_data, "background_color", 2));

  return {samples, depth, width, height, background};
}

inline camera
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

  return camera (lookfrom, lookat, up_vector, fov, aspect_ratio, aperture, dist_to_focus);
}

inline auto
loadScene (const toml::value& scene_data) -> scene
{
  // declare temporary unordered map along with world
  std::unordered_map<string, material> material_map;
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

    auto id = toml::find<string> (mat, "id");
    switch (type[0])
    {
      case 'l':
        material_map.emplace (id, lambertian (c));
        break;
      case 'm':
        material_map.emplace (id, metal (c, toml::find<num> (mat, "fuzz")));
        break;
      case 'd':
        material_map.emplace (id, dielectric (c, toml::find<num> (mat, "ir")));
        break;
      case 'e':
        material_map.emplace (id, emissive (c, toml::find<num> (mat, "intensity")));
        break;
    }
  }

  // Converting unordered map into scene array
  world.material_count = material_map.size ();
  world.materials = new material[world.material_count];
  int i = 0;
  for (const auto& [key, value] : material_map)
  {
    world.materials[i] = value;
    ++i;
  }

  const auto& object_data = toml::find (scene_data, "objects").as_array ();

  world.hittables_size = 2 * object_data.size () - 1;
  world.object_count = object_data.size();
  world.hittables = new hittable[world.hittables_size];

  i = object_data.size() - 1;
  for (const auto& obj : object_data)
  {
    // Referencing material_map to find correct index.
    std::string key = toml::find<std::string> (obj, "material");
    int mat_index =
      std::distance (std::begin (material_map), material_map.find (key));

    std::string geo = toml::find<std::string> (obj, "geometry");

    world.hittables[i] = [&] () -> hittable {
      switch (geo[0])
      {
        case 's': // Sphere
        {
          point3 p (toml::find<num> (obj, "position", 0),
                    toml::find<num> (obj, "position", 1),
                    toml::find<num> (obj, "position", 2));
          return {sphere (p, toml::find<num> (obj, "radius"), mat_index)};
        }
        case 'p': // Plane
        {
          point3 p (toml::find<num> (obj, "position", 0),
                    toml::find<num> (obj, "position", 1),
                    toml::find<num> (obj, "position", 2));
          vec3 n (toml::find<num> (obj, "normal", 0),
                  toml::find<num> (obj, "normal", 1),
                  toml::find<num> (obj, "normal", 2));
          return {plane (p, n, mat_index)};
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
          return {triangle (p1, p2, p3, mat_index)};
        }
        default:
          return {};
      }
    }();
    ++i;
  }

  RandomStateCPU cpu_state;
  RandomState* state = (RandomState*)&cpu_state;
  int start = world.object_count - 1;
  int end = world.hittables_size;

  hittable** hittable_ptrs = new hittable*[world.hittables_size];
  for (int i = 0; i < world.hittables_size; ++i)
  {
    hittable_ptrs[i] = world.hittables + i;
  }

  std::cout << "TREE CONTENTS" << std::endl;
  bounding_tree_node* tree = new bounding_tree_node_node(hittable_ptrs, world.hittables_size, state, start, end);

  std::cout << "ARRAY-TREE CONTENTS" << std::endl;
  convert_tree_to_array(tree, world.hittables);

  std::cout << "FINAL ARRAY CONTENTS" << std::endl;
  for(int i = 0 ; i < world.hittables_size; ++i)
  {
    //bounding_array_node n = world.hittables[i];
    switch (world.hittables[i].id) {
    case hittable_id::Sphere:
      std::cerr << "Sphere " << std::endl;
      break;
    case hittable_id::Plane:
      std::cerr << "Plane " << std::endl;
      break;
    case hittable_id::Rectangle:
      std::cerr << "Rectangle " << std::endl;
      break;
    case hittable_id::Triangle:
      std::cerr << "Triangle " << std::endl;
      break;
    case hittable_id::BoundingBox:
      std::cerr << "BoundingBox " << std::endl;
      break;
    case hittable_id::BoundingArrayNode:
      //std::cerr << "BoundingArrayNode " << n.left << " " << n.right << std::endl;
      std::cerr << "BoundingArrayNode " << std::endl;
      break;
    case hittable_id::Unknown:
      std::cerr << "Unknown " << std::endl;
      break;
    }
  }
  return world;
}

#endif
