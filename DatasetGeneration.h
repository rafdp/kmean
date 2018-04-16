#include "Builder.h"

void GenerateSingleCluster (curandState* states, 
                            const int dimension, 
                            float* mean,  
                            const float dev, 
                            const int Npoints,  
                            float* data,
			    bool* initSetting);

void GenerateDatasetGaussian (const int Npoints, 
                              const int Nclusters, 
                              const int dimension, 
                              float* data, 
                              int* labels,
                              bool shuffle = false,
                              const float stddev = -1.0f);

