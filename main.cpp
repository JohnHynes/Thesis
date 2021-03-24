#include <fstream>
#include <glm/glm.hpp>
#include <iostream>
#include <omp.h>
#include <toml.hpp>

#include "Hittable.hpp"
#include "constants.hpp"
#include "types.hpp"

#include "Camera.hpp"
#include "Image.hpp"
#include "Material.hpp"
#include "Ray.hpp"
#include "TOMLLoader.hpp"
#include "Util.hpp"

__host__ __device__ bool find_closest_hit(const scene *world, ray &r, num t_min,
                                          num t_max, hit_record &hitrec) {
  // Allocate thread-local stack
  using node_ptr = hittable *;
  node_ptr stack[32];
  node_ptr *stack_ptr = stack;

  // Initialize stack
  *stack_ptr++ = NULL;

  // Initialize local variables
  hit_record temp_hitrec;
  num closest_seen = t_max;
  bool has_hit = false;

  // Traverse tree starting from the root
  node_ptr node = world->hittables;
  do {
    if (node->hit(r, t_min, closest_seen, temp_hitrec)) {
      // node was hit, test for leaf
      if (node->id != hittable_id::BoundingArrayNode) {
        // node is a leaf
        if (temp_hitrec.t < closest_seen) {
          closest_seen = temp_hitrec.t;
          hitrec = temp_hitrec;
        }
        has_hit = true;
      } else {
        // node is not a leaf, push left and right children onto stack.
        *stack_ptr++ = world->hittables + node->as_bounding_array_node().left;
        *stack_ptr++ = world->hittables + node->as_bounding_array_node().right;
      }
    }
    // pop node off stack
    node = *--stack_ptr;

  } while (node != NULL);

  return has_hit;
}

color trace_ray(RandomState *state, ray r, color background_color,
                const scene *world, int depth) {
  hit_record rec;
  color attenuation;
  color result_color(1, 1, 1);

  while (depth > 0) {
    // Test bvh for a hit
    if (find_closest_hit(world, r, 0.0001f, infinity, rec)) {
      if (world->materials[rec.mat_idx].scatter((RandomState *)state, r, rec,
                                                attenuation, r)) {
        result_color *= attenuation;
      } else {
        result_color *= world->materials[rec.mat_idx].emit();
        break;
      }
    } else {
      result_color *= background_color;
      break;
    }
    --depth;
  }
  return result_color;
}

int main(int argc, char *argv[]) {
  string filename{"scene.toml"};
  string output{"image.ppm"};
  if (argc > 1) {
    filename = argv[1];
  }
  if (argc > 2) {
    output = argv[2];
  }
  std::cerr << "Command: " << argv[0] << ' ' << filename << ' ' << output
            << '\n';

  const auto scene_data = toml::parse(filename);

  // Image
  auto [samples_per_pixel, max_depth, image_width, image_height,
        background_color] = loadParams(scene_data);

  // World
  scene world = loadScene(scene_data);

  // Camera
  camera cam = loadCamera(scene_data);

  auto image = make_image<color>(image_width, image_height);

  RandomStateCPU cpu_state;
  RandomState *state = (RandomState *)&cpu_state;

  auto start = omp_get_wtime();

  // Rendering image
  #pragma omp parallel for collapse(2) schedule(guided, 16)
  for (int j = 0; j < image_height; ++j) {
    for (int i = 0; i < image_width; ++i) {
      color pixel_color(0.0, 0.0, 0.0);
      for (int s = 0; s < samples_per_pixel; ++s) {
        num u = (i + random_positive_unit(state)) / (image_width - 1);
        num v = (j + random_positive_unit(state)) / (image_height - 1);
        ray r = cam.get_ray(state, u, v);
        pixel_color += trace_ray(state, r, background_color, &world, max_depth);
      }
      image(j, i) = pixel_color;
    }
  }

  auto stop = omp_get_wtime();
  std::cout << (stop - start) << std::endl;

  std::ofstream ofs{output};
  // Outputting Render Data
  ofs << "P3\n";
  ofs << image_width << " " << image_height << "\n";
  ofs << 255 << "\n";

  for (int j = image_height - 1; j >= 0; --j) {
    for (int i = 0; i < image_width; ++i) {
      auto pixel_color = image(j, i);
      write_color(ofs, pixel_color, samples_per_pixel);
    }
  }
}
