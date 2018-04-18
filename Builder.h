#ifndef BUILDER_H_INCLUDED
#define BUILDER_H_INCLUDED

#include <cuda.h>
#include <curand_kernel.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/execution_policy.h>
#include <thrust/binary_search.h>
#include <thrust/functional.h>

#define CC(ans) { GPUAssert((ans), __FILE__, __LINE__); }


void GPUAssert(cudaError_t code, 
                      const char *file, 
                      int line, 
                      bool abort = true);

#define KMEAN_DIMENSION_DEFINED 2
#endif
 
