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
                                float* data, 
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
    data[id] = value*dev + mean[id % dimension];
}

static __global__ void generateUniform(curandState* globalState, 
                                       float* data, 
                                       const float* boundingBox, 
                                       const int maxI, 
                                       const int dimension) 
{
    int id = blockIdx.x *blockDim.x + threadIdx.x;
    if (id >= maxI) return;
    int dim = id % dimension;
    curandState localState = globalState[id];
    float value = curand_uniform ( &localState );
    globalState[id] = localState; 
    data[id] = value*(boundingBox[dimension + dim] - boundingBox[dim]) + boundingBox[dim];
}

/*
static __global__ void setLabels (int* labels)
{
    int cluster = blockIdx.x;
    int point = threadIdx.x;
    labels[point + cluster*blockDim.x] = cluster;
}
*/

void GenerateSingleCluster (curandState* states,
	                    const int seed,	
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
        setup_kernel <<<blocks, kernelsPerBlock>>> (states, seed, Npoints*dimension);
        *initSetting = true;
    }
    generate <<<blocks, kernelsPerBlock>>> (states, data, mean, dev, Npoints*dimension, dimension);
}

void GenerateUniformBox (const int seed,
                         const int dimension,
			 float* boundingBox,
			 const int Npoints,
			 float* data)
{
    curandState* states = nullptr;
    CC(cudaMalloc (&states, Npoints*dimension*sizeof (curandState)));
    
    
    const int kernelsPerBlock = 512;
    const int blocks = Npoints*dimension/kernelsPerBlock + 1;
    setup_kernel <<<blocks, kernelsPerBlock>>> (states, seed, Npoints*dimension);
    
    generateUniform <<<blocks, kernelsPerBlock>>> (states, 
		                                   data, boundingBox, 
						   Npoints*dimension, 
						   dimension);
    
    CC(cudaFree (states));
}


void GenerateDatasetGaussian (const int seed,
		              const int Npoints, 
                              const int Nclusters, 
                              const int dimension, 
                              float* data, 
                              int* labels,
                              bool shuffle,
                              const float stddev)
{
    srand (seed);
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
        GenerateSingleCluster (states, 
			       seed,
			       dimension, 
                               meanD.data().get(), 
                               dev, Npoints, 
                               data + cluster*Npoints*dimension,
			       &initSetting);
    }
    if (shuffle)
    {
        thrust::device_vector<float> swapPoint (dimension);
        float* swapPtr = swapPoint.data().get();
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
 

        }
    }
    CC(cudaFree (states));
}


