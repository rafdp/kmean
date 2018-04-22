
#include "DatasetGeneration.h"
#ifndef KMEAN_DIMENSION_DEFINED 
#error "kmean dimension must be defined"
#endif
class kmean 
{
    const int seed;
    const int K;
    const int Npoints;
    const int Nclusters;
    const int dimension;

    thrust::device_vector<float>* dataD;

    thrust::device_vector<float> centroidsD;
    thrust::device_vector<float> centroidVariancesD;

    thrust::device_vector<int> labelsD;

    int iteration;
public:
    void Iteration ();
    void CentroidInitialization (int seed);
//public:
    kmean (const int seed,
	   const int K,
           const int Npoints,
	   const int Nclusters,
	   const int dimension,
	   thrust::device_vector<float>* dataD_);
    ~kmean ();
    void Process (const int max_iter);

    void Write (std::string filenamePoints, std::string filenameCentroids);
    float Loss ();
};

struct CentroidPointData
{
    float distance;
    int centroid;
    __device__
    bool operator > (const CentroidPointData that) const;
    __device__
    bool operator < (const CentroidPointData that) const;
};

struct LabelAssignmentFunctor
{
    float* dataD;
    float* centroidsD;
    int* labelsD;
    const int dimension;
    const int K;
    LabelAssignmentFunctor (float* dataD_,
		            float* centroidsD_,
			    int* labelsD_,
			    const int dimension_,
			    const int K);
    __device__
    void operator () (int pointIndex);
};


struct DistancePointToCentroidFunctor
{
    float* pointD;
    float* centroidsD;
    const int dimension;
    __device__    
    DistancePointToCentroidFunctor (float* pointD_,
                                    float* centroidsD_,
				    const int dimension_);
    __device__ 
    CentroidPointData operator () (int centroidIndex);

};

struct IteratorSizeHelper
{
    float data[KMEAN_DIMENSION_DEFINED];
    __device__
    IteratorSizeHelper operator+ (const IteratorSizeHelper b) const;
    __device__
    IteratorSizeHelper& operator= (IteratorSizeHelper b);
    __device__
    IteratorSizeHelper operator- (const IteratorSizeHelper b) const;
    __device__
    float NormSq () const;
};

struct CentroidDividerFunctor
{
    float* centroidsD;
    int* keySizesD;
    const int dimension;

    __host__ 
    CentroidDividerFunctor (float* centroidsD_,
		            int* keySizesD_,
			    const int dimension_);

    __device__
    void operator () (int index);
};
struct SquaredNormFunctor
{
    float* dataD;
    float* centroidD;
    const int dimension;
    __device__
    SquaredNormFunctor (float* dataD_,
		       float* centroidD_,
		       const int dimension_);
    __device__
    float operator () (int pointIdx);
};
struct VarianceCalculatorFunctor
{
    float* dataD;
    float* centroidsD;
    float* centroidVariancesD;
    int* centroidMappingD;
    int* cumulativeCentroidSizesD;
    const int dimension;
    VarianceCalculatorFunctor (float* dataD_,
                               float* centroidsD_,
        		       float* centroidVariancesD_,
			       int* centroidMapping_,
			       int* cumulativeCentroidSizesD_,
			       const int dimension_);

    __device__
    void operator () (int centroidIndex);

};
