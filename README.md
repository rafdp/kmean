# kmean
Kmeans clustering algorithm completely based on C++ CUDA.
Developed using thrust for parallel algorithms and curand for dataset generation.

## Features and implementation
The initial centroid coordinates are chosen randomly not as points of the initial array, but rather from the same area.
This resolves the problem of too near centroid stabilization, but has its downsides.
RSS balancing of clusters is used to reduce the overall RSS, and after trying 
BIC and AIC criteria for optimal number of centroids, the method of [gap statistics](http://www.web.stanford.edu/~hastie/Papers/gap.pdf)
was used.
Being not time optimal, as it needs the creation of a uniform distribution over the same area and rerunning kmeans on it multiple times, 
it still gives the best results and is intuitive, opposed to the tested criteria.


[Presentation](https://www.dropbox.com/s/dy0gh3gu3p2y2o4/kmeans.pdf?dl=0)
