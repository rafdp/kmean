#include "kmean.h"


int main( int argc, char** argv) 
{
    if (argc != 4)
    {
        printf ("usage: main [maxSeed] [nPoints] [nIter]\n");
	return 0;
    }
    const int Npoints = atoi (argv[2]);
    const int Nclusters = 5;
    const int Niter = atoi(argv[3]);
    const int NTotal = Npoints*Nclusters;
    const int dimension = KMEAN_DIMENSION_DEFINED;
    const int MaxSeed = atoi (argv[1]);
/*    thrust::device_vector<int> dataLabels (Npoints * Nclusters, 0);
    thrust::device_vector<float> dataD (Npoints * Nclusters * dimension, 0.0f);
    GenerateDatasetGaussian (Npoints, 
                             Nclusters, 
                             dimension, 
                             dataD.data().get (),
                             dataLabels.data().get(),
                             true);
 
    thrust::host_vector<float> dataH (dataD);
    thrust::host_vector<int> labelsH (dataLabels);
    FILE* f = fopen ("data.txt", "w");
    for (int i = 0; i < Npoints*Nclusters; i++)
    {
        for (int d = 0; d < dimension; d++)
            fprintf (f, "%f ", 
                     dataH[i*dimension + d]);
        fprintf (f, "%d \n", labelsH [i]);
    }

    fclose (f);*/
    for (int seed_ = 1; seed_ <= MaxSeed; seed_++)
    {
	std::string mkdirStr ("mkdir data/");
	mkdirStr += std::to_string (seed_);
	system (mkdirStr.c_str ());
	std::string dataFilename ("data/");
	dataFilename += std::to_string (seed_);
	dataFilename += "/loss.txt";
        FILE* loss_file = fopen (dataFilename.c_str (), "w");
	if (!loss_file) return 1;

	if (!(seed_ % 50)) printf ("seed = %d\n", seed_);
	for (int k = 1; k <= 2*Nclusters; k++)
	{

            printf ("seed = %d/%d, k = %d/%d\n", seed_, MaxSeed, k, 2*Nclusters);
            kmean proc (seed_, k, Npoints, Nclusters, dimension);
	    float minLoss = 100.0f;
	    for (int centroidSeed = 1; centroidSeed < 100; centroidSeed++)
	    {
                proc.CentroidInitialization (centroidSeed);
	        for (int iter = 0; iter < Niter; iter++) proc.Iteration();
	        float loss = proc.Loss();
	        if (loss < minLoss) minLoss = loss;
	    }
	    fprintf (loss_file, "%d %f %d %f\n", 
	             k, 
		     minLoss, 
		     2*k*dimension,
		     k*log(dimension*1.0f));
	}
	fclose (loss_file);
    }
    return 0;
}

