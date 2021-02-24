#ifndef GPU_RAY_TRACING_PREPROCESSOR_HPP_
#define GPU_RAY_TRACING_PREPROCESSOR_HPP_

#if defined(USE_GPU) || defined(__CUDACC__)
#include <cuda.h>
#include <stdlib.h>
#define CUDA_CALL(x) do { if((x) != cudaSuccess) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__); \
    exit(-1);}} while(0)
#endif

#if !defined(__host__)
#define __host__
#endif

#if !defined(__device__)
#define __device__
#endif

#endif
