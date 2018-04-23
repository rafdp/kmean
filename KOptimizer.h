

#include "kmean.h"


#include "kmean_cpu.cpp"



class KOptimizer 
{
    const int seed;
    const int Npoints;
    const int Nclusters;
    const int dimension;
    const int Ngap;
    const int maxIter;
    const int maxCentroidSeed;
    thrust::device_vector<float> dataD;
    thrust::device_vector<float> dataUniformD;
    thrust::device_vector<int> initialLabelsD;
    thrust::host_vector<float> RSSkmean;
    thrust::host_vector<float> RSSgapstat;
    //thrust::host_vector<float> gapStatStddev;
    thrust::device_vector<float> boundingBox;
public:
    KOptimizer (const int seed,
		const int Npoints,
		const int Nclusters,
		const int dimension,
		const int Ngap = 10,
		const int maxIter = 40,
		const int maxCentroidSeed = 40);
    ~KOptimizer ();
    float TimeComparison (); 
    void EvaluateKMeans (); 
    void EvaluateGapStatistics ();
    void WriteFileTest ();
};

struct SingleDimensionComparer
{
    int dim;
    __device__
    bool operator () (const IteratorSizeHelper& a, const IteratorSizeHelper& b);

};

struct BoundingDimensionFunctor
{
    IteratorSizeHelper* dataD;
    float* boundingBox;
    const int Ntotal;
    const int dimension;

    __host__ 
    BoundingDimensionFunctor (IteratorSizeHelper* dataD_,
		              float* boundingBox_,
			      const int Ntotal_,
			      const int dimension_);

    __device__ 
    void operator () (int dim); 
};

