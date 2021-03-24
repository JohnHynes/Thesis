#include <fstream>
#include <iostream>

#include "constants.hpp"
#include "types.hpp"

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

__host__ __device__ color trace_ray(RandomState *state, ray r, color background_color,
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
  local_world.hittables_size = world->hittables_size;
  local_world.materials = (material*)shm;
  local_world.hittables = (hittable*)(local_world.materials + local_world.material_count);

  const int local_id = threadIdx.x + blockDim.x * threadIdx.y;
  if (local_id < local_world.material_count) {
    local_world.materials[local_id] = world->materials[local_id];
  }
  if (local_id < local_world.hittables_size) {
    local_world.hittables[local_id] = world->hittables[local_id];
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
  int shmSize = sizeof(material) * world.material_count + sizeof(hittable) * world.hittables_size;

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
