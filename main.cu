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
    printf ("HELLO\n"); 
    for (int seed_ = 4; seed_ <= 3+MaxSeed; seed_++)
    {
        printf ("HELLO %d\n", seed_); 
	KOptimizer kopt (seed_, Npoints, Nclusters, dimension);
        printf ("HELLO %d\n", seed_); 
	kopt.EvaluateGapStatistics ();
        printf ("HELLO %d\n", seed_); 
	kopt.EvaluateKMeans();
        printf ("HELLO %d\n", seed_); 
	kopt.WriteFileTest ();
    }
    printf ("About to return\n");
    return 0;
}

