#include <fstream>
#include <iostream>

#include "constants.hpp"
#include "types.hpp"

#include "TOMLLoader.hpp"
#include "Util.hpp"


__host__ __device__ color
trace_ray (RandomState* state, ray r, color background_color, const scene* world, int depth)
{
  hit_record rec;
  color attenuation;
  color result_color (1, 1, 1);
  while (depth > 0)
  {
    bool has_hit = false;
    {
      num closest_seen = infinity;
      hit_record temp_hitrec;

      for (int i = 0; i < world->object_count; ++i)
      {
        if (world->objects[i].hit (r, 0.0001f, closest_seen, temp_hitrec))
        {
          has_hit = true;
          closest_seen = temp_hitrec.t;
          rec = temp_hitrec;
        }
      }
    }
    if (has_hit)
    {
      if (world->materials[rec.mat_idx].scatter ((RandomState*)state, r, rec, attenuation, r))
      {
        result_color *= attenuation;
      }
      else
      {
        result_color *= world->materials[rec.mat_idx].emit ();
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
__launch_bounds__(256, 4) // gimme 1024 threads
make_image (int seed, int samples_per_pixel, color background_color,
            scene* world, camera* cam, int max_depth, color* d_image)
{
  extern __shared__ char shm[];


  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  const int width = blockDim.x * gridDim.x;
  const int height = blockDim.y * gridDim.y;

  const int id = i + width * j;


  RandomStateGPU state;
  curand_init (seed, id, 0, &state);
  RandomState* rngState = (RandomState*)&state;

  // shared memory

  scene local_world;
  local_world.material_count = world->material_count;
  local_world.object_count = world->object_count;
  local_world.materials = (material*)shm;
  local_world.objects = (hittable*)(local_world.materials + local_world.material_count);

  const int local_id = threadIdx.x + blockDim.x * threadIdx.y;
  if (local_id < local_world.material_count) {
    local_world.materials[local_id] = world->materials[local_id];
  }
  if (local_id < local_world.object_count) {
    local_world.objects[local_id] = world->objects[local_id];
  }

  __syncthreads(); // all threads must wait until the information has been loaded

  color pixel_color (0.0f, 0.0f, 0.0f);
  for (int s = 0; s < samples_per_pixel; ++s)
  {
    num u = (i + random_positive_unit (rngState)) / (width - 1);
    num v = (j + random_positive_unit (rngState)) / (height - 1);
    ray r = cam->get_ray (rngState, u, v);
    pixel_color += trace_ray (rngState, r, background_color, &local_world, max_depth);
  }
  d_image[j * width + i] = pixel_color;
}

int
main (int argc, char* argv[])
{
  string filename{"scene.toml"};
  string output{"image.ppm"};
  int seed{1337};
  if (argc > 1)
  {
    filename = argv[1];
  }
  if (argc > 2)
  {
    output = argv[2];
  }
  if (argc > 3)
  {
    seed = atoi(argv[3]);
  }

  std::cerr << "Command: " << argv[0] << ' ' << filename << ' ' << output << ' ' << seed << '\n';

  const auto scene_data = toml::parse (filename);
  auto [samples_per_pixel, max_depth, image_width, image_height, background_color] = loadParams (scene_data);
  scene world = loadScene (scene_data);
  camera cam = loadCamera (scene_data);

  scene* d_world = world.copy_to_device ();
  camera* d_cam = cam.copy_to_device ();

  color *image, *d_image;
  int num_pixels = image_width * image_height;
  image = new color[num_pixels];
  CUDA_CALL (cudaMalloc ((void**)&d_image, num_pixels * sizeof (color)));

  // Declaring block dimensions
  dim3 threads{16, 16};
  dim3 blocks{image_width / threads.x, image_height / threads.y};
  // request enough shared memory to hold all of the materials and hittables
  int shmSize = sizeof(material) * world.material_count + sizeof(hittable) * world.object_count;

  // Rendering Image on device
  make_image<<<blocks, threads, shmSize>>> (seed, samples_per_pixel, background_color, d_world, d_cam, max_depth, d_image);
  CUDA_CALL (cudaDeviceSynchronize ());

  //d_world->free_device();
  //CUDA_CALL (cudaFree(d_cam));

  // Copying 2D buffer from device to host
  CUDA_CALL (cudaMemcpy (image, d_image, num_pixels * sizeof (color), cudaMemcpyDeviceToHost));
  //CUDA_CALL (cudaFree(d_image));

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

  delete[] image;
}
