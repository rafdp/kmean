#include "Builder.h"

static __global__ void setup_kernel (curandState * state, 
                                     unsigned long seed, 
                                     const int maxI )
{
    int id = blockIdx.x *blockDim.x + threadIdx.x;
    if (id >= maxI) return;
    curand_init ( seed, id, 0, &state[id] );
} 

static __global__ void generate(curandState* globalState, 
                                float* ptr, 
                                const float* mean, 
                                const float dev, 
                                const int maxI, 
                                const int dimension) 
{
    int id = blockIdx.x *blockDim.x + threadIdx.x;
    if (id >= maxI) return;
    curandState localState = globalState[id];
    float value = curand_normal ( &localState );
    globalState[id] = localState; 
    ptr[id] = value*dev + mean[id % dimension];
}

static __global__ void setLabels (int* labels)
{
    int cluster = blockIdx.x;
    int point = threadIdx.x;
    labels[point + cluster*blockDim.x] = cluster;
}


void GenerateSingleCluster (curandState* states, 
                            const int dimension, 
                            float* mean,  
                            const float dev, 
                            const int Npoints,  
                            float* data,
			    bool* initSetting)
{
    const int kernelsPerBlock = 512;
    const int blocks = Npoints*dimension/kernelsPerBlock + 1;
    if (!(*initSetting))
    {
        setup_kernel <<<blocks, kernelsPerBlock>>> (states, time(NULL), Npoints*dimension);
        *initSetting = true;
    }
    generate <<<blocks, kernelsPerBlock>>> (states, data, mean, dev, Npoints*dimension, dimension);
}


void GenerateDatasetGaussian (const int Npoints, 
                              const int Nclusters, 
                              const int dimension, 
                              float* data, 
                              int* labels,
                              bool shuffle,
                              const float stddev)
{
    srand (time(NULL));
    curandState* states = nullptr;
    CC(cudaMalloc (&states, Npoints*dimension*sizeof (curandState)));

    const float dev = (stddev < 0.001f) ? 0.2f/Nclusters : stddev;
    bool initSetting = false;
    for (int cluster = 0; cluster < Nclusters; cluster++)
    { 
        thrust::host_vector<float> mean (dimension, 0.0f);
        for (int d = 0; d < dimension; d++)
            mean[d] = ((rand()*1.0f) /RAND_MAX) * (1-6*dev) + 3*dev;
        thrust::device_vector<float> meanD (mean);
        GenerateSingleCluster (states, dimension, 
                               meanD.data().get(), 
                               dev, Npoints, 
                               data + cluster*Npoints*dimension,
			       &initSetting);
    }
    setLabels <<<Nclusters, Npoints>>> (labels);
    if (shuffle)
    {
        thrust::device_vector<float> swapPoint (dimension);
        float* swapPtr = swapPoint.data().get();
        float swapLabel = 0.0f;
        for (int s = 0; s < Npoints*Nclusters; s++)
        {
            int x = rand () % Npoints*Nclusters;
            int y = s;//rand () % Npoints*Nclusters;
            if (x == y) continue;
            CC(cudaMemcpy (swapPtr, 
                           data+x*dimension, 
                           sizeof(float)*dimension,
                           cudaMemcpyDeviceToDevice));
 
            CC(cudaMemcpy (data+x*dimension, 
                           data+y*dimension, 
                           sizeof(float)*dimension,
                           cudaMemcpyDeviceToDevice));
            
            CC(cudaMemcpy (data+y*dimension, 
                           swapPtr, 
                           sizeof(float)*dimension,
                           cudaMemcpyDeviceToDevice));
            CC(cudaMemcpy (&swapLabel, 
                           labels+x, 
                           sizeof(float),
                           cudaMemcpyDeviceToHost));
 
            CC(cudaMemcpy (labels + x, 
                           labels + y, 
                           sizeof(float),
                           cudaMemcpyDeviceToDevice));

            CC(cudaMemcpy (labels + y, 
                           &swapLabel, 
                           sizeof(float),
                           cudaMemcpyHostToDevice));
        }
    }
    CC(cudaFree (states));
}


