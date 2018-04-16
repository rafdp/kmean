#include "kmean.h"


int main( int argc, char** argv) 
{
    const int Npoints = 100;
    const int Nclusters = 5;
    const int dimension = 2;
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
    printf ("Hello there\n");
    kmean proc (Nclusters, Npoints, Nclusters, dimension);
    printf ("General Kenoby\n");
    proc.Process ();

    return 0;
}

