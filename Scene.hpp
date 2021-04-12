#ifndef GPU_RAY_TRACING_SCENE_HPP_
#define GPU_RAY_TRACING_SCENE_HPP_

#include "Hittable.hpp"
#include "Material.hpp"
#include "Preprocessor.hpp"
#include "types.hpp"

class scene {
public:
  hittable *hittables;
  int hittables_size;
  int object_count;
  material *materials;
  int material_count;

public:
  void free_host() {
    delete hittables;
    delete materials;
  }

#ifdef USE_GPU

  void free_device() {
    cudaFree(hittables);
    cudaFree(materials);
  }

  scene *copy_to_device() {
    scene gpu_scene(*this);
    {
      const int size = sizeof(hittable) * hittables_size;
      CUDA_CALL(cudaMalloc(&gpu_scene.hittables, size));
      CUDA_CALL(
          cudaMemcpy(gpu_scene.hittables, hittables, size, cudaMemcpyHostToDevice));
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
