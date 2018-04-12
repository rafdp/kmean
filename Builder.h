#ifndef BUILDER_H_INCLUDED
#define BUILDER_H_INCLUDED

#include <cuda.h>
#include <curand_kernel.h>
#include <thrust/device_vector.h>

#define CC(ans) { GPUAssert((ans), __FILE__, __LINE__); }


void GPUAssert(cudaError_t code, 
                      const char *file, 
                      int line, 
                      bool abort = true);


#endif
 
