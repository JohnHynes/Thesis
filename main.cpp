#include <iostream>
#include <glm/glm.hpp>
#include <toml.hpp>
#include <fstream>
#include <omp.h>

#include "types.hpp"
#include "constants.hpp"

#include "Image.hpp"
#include "Camera.hpp"
#include "Ray.hpp"
#include "Util.hpp"
#include "Material.hpp"
#include "TOMLLoader.hpp"

color
trace_ray (RandomState* state, ray r, const scene* world, int depth)
{

  hit_record rec;
  color attenuation;
  color factor (1, 1, 1);
  color result_color = factor;

  while (depth > 0)
  {
    bool has_hit = false;
    {
      num closest_seen = infinity;
      hit_record temp_hitrec;

      for (int i = 0; i < world->object_count; ++i)
      {
        if (world->objects[i]->hit (r, 0.0001, closest_seen, temp_hitrec))
        {
          has_hit = true;
          closest_seen = temp_hitrec.t;
          rec = temp_hitrec;
        }
      }
    }
    if (has_hit)
    {
      if (world->materials[rec.mat_idx]->scatter ((RandomState*)state, r, rec,
                                                  attenuation, r))
      {
        factor *= attenuation;
      }
      else
      {
        result_color = color (0, 0, 0);
        break;
      }
    }
    else
    {
      num t = 0.5 * (glm::normalize (r.dir).y + 1.0);
      result_color = glm::mix (color (1.0, 1.0, 1.0), color (0.5, 0.7, 1.0), t);
      break;
    }
    --depth;
  }
  return result_color * factor;
}

int main(int argc, char* argv[])
{
 string filename{"scene.toml"};
  string output{"image.ppm"};
  if (argc > 1)
  {
    filename = argv[1];
  }
  if (argc > 2)
  {
    output = argv[2];
  }
  std::cerr << "Command: " << argv[0] << ' ' << filename << ' ' << output
            << '\n';

    const auto scene_data = toml::parse(filename);

    // Image
    auto [samples_per_pixel, max_depth, image_width, image_height] = loadParams(scene_data);

    // World
    scene world = loadScene(scene_data);

    // Camera
    camera cam = loadCamera(scene_data);

    auto image = make_image<color>(image_width, image_height);

    RandomStateCPU cpu_state;
    RandomState* state = (RandomState*)&cpu_state;

    auto start = omp_get_wtime();

    // Rendering image
    #pragma omp parallel for collapse(2) schedule(guided, 16)
    for (int j = 0; j < image_height; ++j)
    {
        for (int i = 0; i < image_width; ++i)
        {
            color pixel_color(0.0, 0.0, 0.0);
            for (int s = 0; s < samples_per_pixel; ++s)
            {
                num u = (i + random_positive_unit(state)) / (image_width-1);
                num v = (j + random_positive_unit(state)) / (image_height-1);
                ray r = cam.get_ray(state, u, v);
                pixel_color += trace_ray(state, r, &world, max_depth);
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

  for (int j = image_height - 1; j >= 0; --j)
  {
    for (int i = 0; i < image_width; ++i)
    {
      auto pixel_color = image(j, i);
      write_color (ofs, pixel_color, samples_per_pixel);
    }
  }
}
