
#include "KOptimizer.h"


KOptimizer::KOptimizer (const int seed_,
		        const int Npoints_,
			const int Nclusters_,
			const int dimension_, 
			const int Ngap_,
			const int maxIter_,
			const int maxCentroidSeed_) : 
    seed            (seed_),
    Npoints         (Npoints_),
    Nclusters       (Nclusters_),
    dimension       (dimension_),
    Ngap            (Ngap_),
    maxIter         (maxIter_),
    maxCentroidSeed (maxCentroidSeed_),
    dataD           (Npoints*Nclusters*dimension, 0.0f),
    dataUniformD    (Npoints*Nclusters*dimension, 0.0f),
    initialLabelsD  (Npoints*Nclusters),
    RSSkmean        (2*Nclusters),
    RSSgapstat      (2*Nclusters),
    //gapStatStddev   (2*Nclusters),
    boundingBox     (2*dimension, 0.0f)
{
    GenerateDatasetGaussian (seed, 
		             Npoints, 
		             Nclusters,
			     dimension,
			     dataD.data().get(),
			     initialLabelsD.data().get(),
			     true);
    thrust::counting_iterator<int> counter (0);
    BoundingDimensionFunctor bdf ((IteratorSizeHelper*) dataD.data().get(),
		                  boundingBox.data().get(),
				  Npoints*Nclusters,
				  dimension);
    thrust::for_each (thrust::device,
		      counter,
		      counter + dimension,
		      bdf);
}
KOptimizer::~KOptimizer () {}

float KOptimizer::TimeComparison ()
{
    kmean proc (seed, Nclusters, Npoints, Nclusters, dimension, dataD.data().get());
    proc.CentroidInitialization (seed);
    timespec timeGPU0 = {};
    timespec timeGPU1 = {};
    timespec timeCPU0 = {};
    timespec timeCPU1 = {};
    clock_gettime(CLOCK_REALTIME, &timeGPU0);
    for (int i = 0; i < maxIter; i++) proc.Iteration ();
    clock_gettime(CLOCK_REALTIME, &timeGPU1);
     thrust::host_vector<double> dataH (dataD);
    vector<double> dataVector (dataH.begin(), dataH.end());
     vector<Point> points;
    for(int i = 0; i < Npoints*Nclusters; i++)
    {
        vector<double> values (dataH.begin() + i*dimension, dataH.begin() + (i+1)*dimension);
        Point p(i, values);
        points.push_back (p);
    }
    KMeans kmeans(Nclusters, Npoints*Nclusters, dimension, maxIter);
    clock_gettime(CLOCK_REALTIME, &timeCPU0);
    kmeans.run(points);
    clock_gettime(CLOCK_REALTIME, &timeCPU1);
    size_t timeCPU = ((timeCPU1.tv_sec - timeCPU0.tv_sec)*1000000000.0f + 
                       timeCPU1.tv_nsec - timeCPU0.tv_nsec)/1000.0f;
    size_t timeGPU = ((timeGPU1.tv_sec - timeGPU0.tv_sec)*1000000000.0f + 
                       timeGPU1.tv_nsec - timeGPU0.tv_nsec)/1000.0f;
    return timeCPU*1.0f/timeGPU;
}

void KOptimizer::EvaluateKMeans ()
{
    for (int k = 1; k <= 2*Nclusters; k++)
    {
        //printf ("seed = %d/%d, k = %d/%d\n", seed_, MaxSeed, k, 2*Nclusters);
        kmean proc (seed, k, Npoints, Nclusters, dimension, dataD.data().get());
	__
        float minLoss = 10000.0f;
        for (int centroidSeed = 1; centroidSeed < maxCentroidSeed; centroidSeed++)
        {
            proc.CentroidInitialization (centroidSeed);
            for (int iter = 0; iter < maxIter; iter++) proc.Iteration();
            float loss = proc.Loss();
            if (loss < minLoss) minLoss = loss;
        }
	__
	RSSkmean[k-1] = log(minLoss);
	printf ("k = %d, minRss = %f\n", k, minLoss);
    }
}

void KOptimizer::EvaluateGapStatistics ()
{

    for (int gapDataset = 0; gapDataset < Ngap; gapDataset++)
    {
        GenerateUniformBox (time(nullptr) + gapDataset,
			    dimension,
			    boundingBox.data().get(),
			    Npoints*Nclusters,
			    dataUniformD.data().get());
	float t = dataUniformD[0];
        printf ("uniform p %f\n", t);
        for (int k = 1; k <= 2*Nclusters; k++)
        {
            kmean proc (seed, k, Npoints, Nclusters, dimension, dataUniformD.data().get());
            proc.CentroidInitialization (time(nullptr) + gapDataset);
            for (int iter = 0; iter < maxIter; iter++) proc.Iteration();
	    RSSgapstat[k-1] += log(proc.Loss ());
        }
	printf ("gap dataset = %d\n", gapDataset);
    }
    for (int gapDataset = 0; gapDataset < Ngap; gapDataset++) RSSgapstat[gapDataset] /= Ngap*1.0f;

    
}

void KOptimizer::WriteFileTest ()
{
    thrust::host_vector<float> dataH (dataD);
    thrust::host_vector<float> dataUniformH (dataUniformD);
    FILE* f_data = fopen ("data.txt", "w");
    for (int i = 0; i < Npoints*Nclusters; i++)
    {
        for (int d = 0; d < dimension; d++)
	    fprintf (f_data, "%f ", dataH[i*dimension + d]);
	fprintf (f_data, " 1\n");
        for (int d = 0; d < dimension; d++)
	    fprintf (f_data, "%f ", dataUniformH[i*dimension + d]);
	fprintf (f_data, " 2\n");
    }
    fclose (f_data);


    FILE* f_logW = fopen ("logRSS.txt", "w");
    for (int k = 1; k < 2*Nclusters; k++)
    {
        fprintf (f_logW, "%d %f %f\n", k, RSSkmean[k-1], RSSgapstat[k-1]);
    }

    fclose (f_logW);

}

__device__
bool SingleDimensionComparer::operator () (const IteratorSizeHelper& a, 		
		                          const IteratorSizeHelper& b)
{
    return a.data[dim] < b.data[dim];
}

__host__ 
BoundingDimensionFunctor::BoundingDimensionFunctor (IteratorSizeHelper* dataD_,
		          float* boundingBox_,
			  const int Ntotal_,
			  const int dimension_) : 
    dataD       (dataD_),
    boundingBox (boundingBox_),
    Ntotal      (Ntotal_),
    dimension   (dimension_)
{}

__device__ 
void BoundingDimensionFunctor::operator () (int dim)
{
    SingleDimensionComparer sdc = {dim};
    auto iter = thrust::minmax_element (thrust::device,
                                        dataD,
			                dataD+Ntotal,
				        sdc);
    boundingBox[dim] = iter.first->data[dim];
    boundingBox[dim+dimension] = iter.second->data[dim];
}


