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
	if (!(seed_ % 50)) printf ("seed = %d\n", seed_);
        kmean proc (seed_, Nclusters, Npoints, Nclusters, dimension);
        proc.CentroidInitialization ();
        for (int i = 0; i < Niter+1; i++)
        {
	    std::string dataFilename ("data/");
	    dataFilename += std::to_string (seed_);
	    dataFilename += "/data_";
	    dataFilename += std::to_string (i);
	    dataFilename += ".txt";

	    std::string centrFilename ("data/");
    	    centrFilename += std::to_string (seed_);
    	    centrFilename += "/centr_";
	    centrFilename += std::to_string (i);
	    centrFilename += ".txt";
            proc.Write (dataFilename, centrFilename);
            proc.Iteration ();
        }
    }
    return 0;
}

