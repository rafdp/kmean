#include "KOptimizer.h"


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
    cudaFree (0);
    __

    FILE* f = fopen ("time.txt", "w");
    if (!f) return 1;
    for (int np = 100; np <= 100; np++)
    {
	/*for (int ni = 1; ni <= Niter; ni++)
        {   printf ("ni = %d\n", ni);
	    KOptimizer kopt (4, np, Nclusters, dimension, 10, ni);
            __
            float comp = kopt.TimeComparison ();
	    fprintf (f, "%d %d %f\n", np, ni, comp);
	}
	    printf ("np = %d\n", np);*/
	KOptimizer kopt (4, Npoints, Nclusters, dimension, 10, 1);
	__
        kopt.TimeComparison ();
	//kopt.EvaluateGapStatistics ();
	//kopt.EvaluateKMeans();
	//kopt.WriteFileTest ();
    }
    fclose (f);
    printf ("About to return\n");
    return 0;
}

