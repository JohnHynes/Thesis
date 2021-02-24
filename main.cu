#include <cuda.h>
#include <cuda_runtime_api.h>

#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

#include <iostream>
#include <toml.hpp>

#include "constants.hpp"
#include "types.hpp"

#include "Camera.hpp"
#include "Material.hpp"
#include "Ray.hpp"
#include "TOMLLoader.hpp"
#include "Util.hpp"

#include <fstream>

__host__ __device__
color
trace_ray (RandomState* state, ray r, color background_color, const scene* world, int depth)
{
  hit_record rec;
  color attenuation;
  color result_color(1, 1, 1);
  while (depth > 0)
  {
    bool has_hit = false;
    {
      num closest_seen = infinity;
      hit_record temp_hitrec;

      for (int i = 0; i < world->object_count; ++i)
      {
        if (world->objects[i]->hit (r, 0.0001f, closest_seen, temp_hitrec))
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
        result_color *= attenuation;
      }
      else
      {
        result_color *= world->materials[rec.mat_idx]->emit();
        break;
      }
    }
    else
    {
      result_color *= background_color;
      break;
    }
    --depth;
  }
  return result_color;
}

__global__ void
make_image (int seed, int samples_per_pixel, color background_color,
            scene* world, camera* cam, int max_depth, color* d_image)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  const int width = blockDim.x * gridDim.x;
  const int height = blockDim.y * gridDim.y;

  const int id = i + width * j;
  RandomStateGPU state;
  curand_init (seed, id, 0, &state);
  RandomState* rngState = (RandomState*)&state;

  color pixel_color (0.0, 0.0, 0.0);
  for (int s = 0; s < samples_per_pixel; ++s)
  {
    num u = (i + random_positive_unit (rngState)) / (width - 1);
    num v = (j + random_positive_unit (rngState)) / (height - 1);
    ray r = cam->get_ray (rngState, u, v);
    pixel_color += trace_ray (rngState, r, background_color, world, max_depth);
  }
  d_image[j * width + i] = pixel_color;
}

int
main (int argc, char* argv[])
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

  // Parsing scene data from a toml file
  const auto scene_data = toml::parse (filename);
  std::cerr << "parse toml\n";

  // Image
  auto [samples_per_pixel, max_depth, image_width, image_height, background_color] =
    loadParams (scene_data);
  std::cerr << "load image\n";

  // World
  scene world = loadScene (scene_data);

  std::cerr << "load scene\n";

  // Copying world to device
  scene* d_world = world.copy_to_device ();

  std::cerr << "copy scene\n";

  // Camera
  camera cam = loadCamera (scene_data);

  std::cerr << "load camera\n";

  // Copying camera to device
  camera* d_cam = cam.copy_to_device ();

  std::cerr << "copy camera\n";

  // Allocating 2D buffer on device
  color *image, *d_image;
  int num_pixels = image_width * image_height;
  image = new color[num_pixels];
  CUDA_CALL (cudaMalloc ((void**)&d_image, num_pixels * sizeof (color)));

  std::cerr << "allocated image on device\n";

  // Declaring block dimensions
  dim3 threads{16, 16};
  dim3 blocks{image_width / threads.x, image_height / threads.y};

  // Rendering Image on device
  for(int i = 0; i < 1; ++i) {
    make_image<<<blocks, threads>>> (1337, samples_per_pixel, background_color, d_world,
                                    d_cam, max_depth, d_image);

    CUDA_CALL (cudaDeviceSynchronize ());

    std::cerr << "called kernel\n";
  }

  // Copying 2D buffer from device to host
  CUDA_CALL (cudaMemcpy (image, d_image, num_pixels * sizeof (color),
                         cudaMemcpyDeviceToHost));

  std::ofstream ofs{output};
  // Outputting Render Data
  ofs << "P3\n";
  ofs << image_width << " " << image_height << "\n";
  ofs << 255 << "\n";

  for (int j = image_height - 1; j >= 0; --j)
  {
    for (int i = 0; i < image_width; ++i)
    {
      auto pixel_color = image[j * image_width + i];
      write_color (ofs, pixel_color, samples_per_pixel);
    }
  }

  // Freeing Memory
  free (image);
  CUDA_CALL (cudaFree (d_image));
}
