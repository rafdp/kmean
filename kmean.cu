
#include "kmean.h"

kmean::kmean (const int K_,
	      const int Npoints_,
	      const int Nclusters_,
	      const int dimension_) : 
    K              (K_),
    Npoints        (Npoints_),
    Nclusters      (Nclusters_),
    dimension      (dimension_),
    dataD          (Npoints*Nclusters*dimension, 0.0f),
    initialLabelsD (Npoints*Nclusters, 0),
    centroidsD     (K*dimension),
    labelsD        (Npoints*Nclusters)
{
    GenerateDatasetGaussian (Npoints, 
		             Nclusters,
			     dimension,
			     dataD.data().get(),
			     initialLabelsD.data().get(),
			     true);

}

void kmean::Write (std::string filenamePoints, std::string filenameCentroids)
{ 
    FILE* f_points = fopen (filenamePoints.c_str (), "w");
    if (!f_points) return;
    FILE* f_centroids = fopen (filenameCentroids.c_str (), "w");
    if (!f_centroids) {fclose (f_points); return;}

    thrust::host_vector<float> dataH (dataD);
    thrust::host_vector<int> labelsH (labelsD);
    thrust::host_vector<int> initialLabelsH (initialLabelsD);
    for (int i = 0; i < Npoints*Nclusters; i++)
    {
        for (int d = 0; d < dimension; d++)
            fprintf (f_points, "%f ", 
                     dataH[i*dimension + d]);
        fprintf (f_points, "%d %d\n", labelsH [i], initialLabelsH [i]);
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


void kmean::CentroidInitialization ()
{
    curandState* states = nullptr;
    CC(cudaMalloc (&states, Npoints*dimension*sizeof (curandState)));
    bool singleClusterInitSetting = false;
    thrust::device_vector<float> centroidsInit (K*dimension, 0.5f);
    GenerateSingleCluster (states, 
		           dimension,
			   centroidsInit.data().get (),
			   0.5f/3.0f,
			   K,
			   centroidsD.data().get(),
			   &singleClusterInitSetting);
    CC(cudaFree (states));
}


void kmean::Iteration ()
{
    LabelAssignmentFunctor laf (dataD.data().get(),
		                centroidsD.data().get(),
			        labelsD.data().get(),
			        dimension,
			        K);
    thrust::counting_iterator<int> pointCounter (0);
    thrust::for_each (thrust::device,
		      pointCounter,
		      pointCounter + Npoints*Nclusters,
		      laf);
}

void kmean::Process ()
{
    printf ("About to call centroid initialization\n");
    CentroidInitialization();
    printf ("About to call iteration\n");
    Iteration ();
    printf ("About to write\n");
    Write ("test_data.txt", "test_centroids.txt");
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
	distance += tempValue*tempValue;
    }
    CentroidPointData cpd = {distance, centroidIndex};
    return cpd;
}



