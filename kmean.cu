
#include "kmean.h"

kmean::kmean (const int seed_,
	      const int K_,
	      const int Npoints_,
	      const int Nclusters_,
	      const int dimension_,
	      thrust::device_vector<float>* dataD_) : 
    seed               (seed_),
    K                  (K_),
    Npoints            (Npoints_),
    Nclusters          (Nclusters_),
    dimension          (dimension_),
    dataD              (dataD_),
    centroidsD         (K*dimension),
    centroidVariancesD (K),
    labelsD            (Npoints*Nclusters, -1),
    iteration          ()
{
    /*GenerateDatasetGaussian (seed, 
		             Npoints, 
		             Nclusters,
			     dimension,
			     dataD.data().get(),
			     initialLabelsD.data().get(),
			     true);*/

}
kmean::~kmean () {}

void kmean::Write (std::string filenamePoints, std::string filenameCentroids)
{ 
    FILE* f_points = fopen (filenamePoints.c_str (), "w");
    if (!f_points) return;
    FILE* f_centroids = fopen (filenameCentroids.c_str (), "w");
    if (!f_centroids) {fclose (f_points); return;}

    thrust::host_vector<float> dataH (*dataD);
    thrust::host_vector<int> labelsH (labelsD);
    for (int i = 0; i < Npoints*Nclusters; i++)
    {
        for (int d = 0; d < dimension; d++)
            fprintf (f_points, "%f ", 
                     dataH[i*dimension + d]);
        fprintf (f_points, "%d\n", labelsH [i]);
    }

    fclose (f_points);
    thrust::host_vector<float> centroidsH (centroidsD);
    for (int c = 0; c < K; c++)
    {
        for (int d = 0; d < dimension; d++)
            fprintf (f_centroids, "%f ", 
                     centroidsH[c*dimension + d]);
        fprintf (f_centroids, "%d\n", c);
    }
    fclose (f_centroids);
}


void kmean::CentroidInitialization (int seed_)
{
    curandState* states = nullptr;
    CC(cudaMalloc (&states, Npoints*dimension*sizeof (curandState)));
    bool singleClusterInitSetting = false;
    thrust::device_vector<float> centroidsInit (K*dimension, 0.5f);
    GenerateSingleCluster (states, 
		           seed_,
		           dimension,
			   centroidsInit.data().get (),
			   0.5f/3.0f,
			   K,
			   centroidsD.data().get(),
			   &singleClusterInitSetting);
    CC(cudaFree (states));
    iteration = 0;
}


void kmean::Iteration ()
{
    LabelAssignmentFunctor laf (dataD->data().get(),
		                centroidsD.data().get(),
			        labelsD.data().get(),
			        dimension,
			        K);
    thrust::counting_iterator<int> pointCounter (0);
    thrust::for_each (thrust::device,
		      pointCounter,
		      pointCounter + Npoints*Nclusters,
		      laf);

    typedef IteratorSizeHelper* CustomPtr;
    
    thrust::device_vector<float> d_centroidsD (centroidsD);
    CustomPtr dataCustomPtr = reinterpret_cast<CustomPtr> (dataD->data().get());
    CustomPtr centroidsCustomPtr = reinterpret_cast<CustomPtr> (centroidsD.data().get());
    CustomPtr d_centroidsCustomPtr = reinterpret_cast<CustomPtr> (d_centroidsD.data().get());

    thrust::sort_by_key (thrust::device,
		         labelsD.begin(),
			 labelsD.end(),
			 dataCustomPtr);

    thrust::device_vector<int> keyDump (K, 0);
    auto ends_pair = thrust::reduce_by_key (thrust::device,
		                            labelsD.begin(),
			                    labelsD.end(),
			                    dataCustomPtr,
			                    keyDump.begin(),
			                    d_centroidsCustomPtr,
			                    thrust::equal_to<int> (),
			                    thrust::plus<IteratorSizeHelper> ());
    int* keyDumpPtr = keyDump.data().get();
    const int usedCentroids = ends_pair.first - keyDump.begin();
    thrust::device_vector<int> clusterSizes (keyDump);
    thrust::upper_bound (thrust::device,
		         labelsD.begin(),
			 labelsD.end(),
			 keyDump.begin(),
			 keyDump.end(),
			 clusterSizes.begin());

    VarianceCalculatorFunctor vcf (dataD->data().get(),
		                   centroidsD.data().get(),
				   centroidVariancesD.data().get(),
				   keyDump.data().get(),
				   clusterSizes.data().get(),
				   dimension);
    centroidVariancesD.assign(K, 0.0f);
    thrust::for_each (thrust::device,
		      pointCounter,
		      pointCounter + K,
		      vcf);
    
    auto minMaxPair = thrust::minmax_element (thrust::device, 
		                              centroidVariancesD.begin(),
					      centroidVariancesD.begin() + usedCentroids);
    int maxCentroidIter = minMaxPair.second - centroidVariancesD.begin();
    int minCentroidIter = minMaxPair.first - centroidVariancesD.begin();
    bool swapMinMax = (*(minMaxPair.second)/(*(minMaxPair.first)) > 10.0f);
    CustomPtr shiftPoint = dataCustomPtr + clusterSizes[maxCentroidIter] -1;
    thrust::adjacent_difference (thrust::device,
		                 clusterSizes.begin(),
				 clusterSizes.end(),
				 clusterSizes.begin());
    CentroidDividerFunctor cdf (d_centroidsD.data().get(),
		                clusterSizes.data().get(),
				dimension);
    thrust::for_each (pointCounter,
		      pointCounter + K,
		      cdf);
    if (iteration >= 5) 
    thrust::for_each (thrust::device,
		      pointCounter,
		      pointCounter + K,
		      [centroidsCustomPtr, 
		       keyDumpPtr, 
		       d_centroidsCustomPtr, 
		       shiftPoint] __device__ (int idx)
		       { 
			  if (keyDumpPtr[idx] != idx && 
			      (!idx || keyDumpPtr[idx - 1] == idx - 1))
			  {centroidsCustomPtr[idx] = *shiftPoint;}
			  else
		          if (keyDumpPtr[idx] >= idx) 
		              centroidsCustomPtr[keyDumpPtr[idx]] = 
		              d_centroidsCustomPtr[idx];
		      });
    else
    thrust::for_each (thrust::device,
		      pointCounter,
		      pointCounter + K,
		      [centroidsCustomPtr, 
		       keyDumpPtr, 
		       d_centroidsCustomPtr] __device__ (int idx)
		       { 
		          if (keyDumpPtr[idx] >= idx) 
		              centroidsCustomPtr[keyDumpPtr[idx]] = 
		              d_centroidsCustomPtr[idx];
		      });
    if (iteration >= 5 && swapMinMax && usedCentroids == K) 
    {
	CustomPtr minPtr = centroidsCustomPtr + keyDump[minCentroidIter];
	CC(cudaMemcpy (minPtr, shiftPoint, sizeof (float)*dimension, cudaMemcpyDeviceToDevice));
	iteration = 0;
    }
    if (usedCentroids != K && iteration >= 5) iteration = 0;
    iteration++;
}


void kmean::Process (const int max_iter)
{
    CentroidInitialization(seed);
    for (int i = 0; i < max_iter; i++) Iteration ();
    //printf ("About to write\n");
    //Write ("test_data.txt", "test_centroids.txt");
}

float kmean::Loss ()
{
    return thrust::reduce (thrust::device,
		           centroidVariancesD.begin (),
			   centroidVariancesD.end ());
}

__device__
bool CentroidPointData::operator > (const CentroidPointData that) const
{
    return distance > that.distance;
}
__device__
bool CentroidPointData::operator < (const CentroidPointData that) const
{
    return distance < that.distance;
}

LabelAssignmentFunctor::LabelAssignmentFunctor (float* dataD_,
		                                float* centroidsD_,
						int* labelsD_,
						const int dimension_, 
						const int K_) : 
    dataD      (dataD_),
    centroidsD (centroidsD_),
    labelsD    (labelsD_),
    dimension  (dimension_),
    K          (K_)
{}
__device__
void LabelAssignmentFunctor::operator () (int pointIndex)
{
    thrust::minimum<CentroidPointData> minimizer;
    thrust::counting_iterator<int> centroidCounter (0);
    DistancePointToCentroidFunctor dptcf (dataD + pointIndex*dimension,
		                          centroidsD,
					  dimension);
    CentroidPointData result = {100.0f, K + 1};
    result = thrust::transform_reduce (thrust::device, 
		                       centroidCounter, 
			               centroidCounter + K,
			               dptcf,
			               result,
			               minimizer);
    labelsD[pointIndex] = result.centroid;
}
__device__
DistancePointToCentroidFunctor::DistancePointToCentroidFunctor (float* pointD_,
                                                                float* centroidsD_,
								const int dimension_) : 
    pointD     (pointD_),
    centroidsD (centroidsD_),
    dimension  (dimension_)
{}
__device__
CentroidPointData DistancePointToCentroidFunctor::operator () (int centroidIndex)
{
    float distance = 0.0f;
    float tempValue = 0.0f;
    for (int d = 0; d < dimension; d++)
    {
	tempValue = (pointD[d] - centroidsD[centroidIndex*dimension + d]);
	distance += sqrt(tempValue*tempValue);
    }
    CentroidPointData cpd = {distance, centroidIndex};
    return cpd;
}

__device__
IteratorSizeHelper IteratorSizeHelper::operator+ (const IteratorSizeHelper b) const
{
    IteratorSizeHelper res = {};
    for (int d = 0; d < KMEAN_DIMENSION_DEFINED; d++)
    {
        res.data[d] = data[d] + b.data[d];
    }
    return res;
}
__device__
IteratorSizeHelper& IteratorSizeHelper::operator= (IteratorSizeHelper b) 
{
    for (int d = 0; d < KMEAN_DIMENSION_DEFINED; d++)
    {
        data[d] = b.data[d];
    }
    return *this;
}
__device__
IteratorSizeHelper IteratorSizeHelper::operator- (const IteratorSizeHelper b) const
{
    IteratorSizeHelper res = {};
    for (int d = 0; d < KMEAN_DIMENSION_DEFINED; d++)
    {
        res.data[d] = data[d] - b.data[d];
    }
    return res;
}
__device__
float IteratorSizeHelper::NormSq () const
{
    float res = 0.0f;
    for (int d = 0; d < KMEAN_DIMENSION_DEFINED; d++)
    {
        res += data[d]*data[d];
    }
    return res;
}


__host__ 
CentroidDividerFunctor::CentroidDividerFunctor (float* centroidsD_,
                        int* keySizesD_,
	                const int dimension_) : 
    centroidsD (centroidsD_),
    keySizesD  (keySizesD_),
    dimension (dimension_)
{}

__device__
void CentroidDividerFunctor::operator () (int index)
{
    if (!keySizesD[index]) return;
    for (int d = 0; d < dimension; d++)
    {
        centroidsD[index*dimension + d] /= keySizesD[index];
    }

}
__device__
SquaredNormFunctor::SquaredNormFunctor (float* dataD_,
	                               float* centroidD_,
				       const int dimension_) :
    dataD     (dataD_),
    centroidD (centroidD_),
    dimension (dimension_)
{}
__device__
float SquaredNormFunctor::operator () (int pointIndex)
{
    float value = dataD[pointIndex] - centroidD[pointIndex % dimension];
    return value*value;
}

VarianceCalculatorFunctor::VarianceCalculatorFunctor (float* dataD_,
                                                      float* centroidsD_,
       		                                      float* centroidVariancesD_,
						      int* centroidMappingD_,
		                                      int* cumulativeCentroidSizesD_,
						      const int dimension_) : 
    dataD                    (dataD_),
    centroidsD               (centroidsD_),
    centroidVariancesD       (centroidVariancesD_),
    centroidMappingD         (centroidMappingD_),
    cumulativeCentroidSizesD (cumulativeCentroidSizesD_),
    dimension                (dimension_)
{}

__device__
void VarianceCalculatorFunctor::operator () (int centroidIndex)
{
    if (centroidMappingD[centroidIndex] < centroidIndex) return;
    SquaredNormFunctor sqf (dataD, centroidsD + centroidMappingD[centroidIndex]*dimension, dimension);

    thrust::counting_iterator<int> counter (0);
    float variance =  
        thrust::transform_reduce (thrust::device,
			          counter + ((centroidIndex > 0) ? 
				           cumulativeCentroidSizesD[centroidIndex - 1] : 
				           0)*dimension,
			          counter + cumulativeCentroidSizesD[centroidIndex]*dimension,
				  sqf,
				  0.0f,
			          thrust::plus<float> ());
    centroidVariancesD[centroidIndex] = variance;
}

