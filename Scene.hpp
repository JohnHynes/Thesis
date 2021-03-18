#ifndef GPU_RAY_TRACING_SCENE_HPP_
#define GPU_RAY_TRACING_SCENE_HPP_

#include "Hittable.hpp"
#include "Material.hpp"
#include "Preprocessor.hpp"
#include "types.hpp"

class scene {
public:
  hittable *objects;
  int object_count;
  //hittable *bvh;
  //int bvh_size;
  material *materials;
  int material_count;

public:
  /*
  __host__ void generate_bvh() {
    // Create modifiable array of objects
    hittable *temp_objects = new hittable[object_count];
    std::copy(objects, objects + object_count, temp_objects);

    bvh = new hittable[object_count - 1];

    for (hittable *h = objects; h < objects + object_count; ++h) {

    }
  }
  */
  void free_host() {
    delete objects;
    //delete bvh;
    delete materials;
  }

#ifdef USE_GPU

  void free_device() {
    cudaFree(objects);
    cudaFree(materials);
  }

  scene *copy_to_device() {
    scene gpu_scene(*this);
    {
      const int size = sizeof(hittable) * object_count;
      CUDA_CALL(cudaMalloc(&gpu_scene.objects, size));
      CUDA_CALL(
          cudaMemcpy(gpu_scene.objects, objects, size, cudaMemcpyHostToDevice));
    }
    {
      const int size = sizeof(material) * material_count;
      CUDA_CALL(cudaMalloc(&gpu_scene.materials, size));
      CUDA_CALL(cudaMemcpy(gpu_scene.materials, materials, size,
                           cudaMemcpyHostToDevice));
    }

    scene *d_scene;
    CUDA_CALL(cudaMalloc(&d_scene, sizeof(scene)));
    CUDA_CALL(
        cudaMemcpy(d_scene, &gpu_scene, sizeof(scene), cudaMemcpyHostToDevice));

    return d_scene;
  }

#endif
};

#endif
