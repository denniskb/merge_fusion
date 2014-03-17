# Accompanying notes for: Reconstructing Rome in Three Minutes

## Benefits of this system over competitors

- Global tracking: Incoming point clouds are registered globally against the entire already reconstructed data set. The bigger search space improves the quality of the solution. It also means that there are no limitations on the movement speed of the user. In case the algorithm is unable to match an incoming point cloud, it creates a separate volume/connected component and fuses multiple volumes once it finds similarities between them. Implicitly handles localization -> no setup required: User can start to (re)scan scene and once a match is fined it gets fused into existing scans. More scans improve reconstruction over time.
- Loop closure: Perfect volumes without the usual accumulating drift. Allows revisiting (pausing/continuing).


## Benefits of using point clouds

- Closer to input data -> no conversion required. Allows simpler/more unified processing of data and simpler algorithm design.
- Requires no visualization for the reconstruction (e.g. depth map extraction), since matching happens directly on point clodus (3D pattern recognition).
- Equivalent to deferring visualization to a later stage. All of this makes the algorithm computationally less expensive and allows its implementation on 'weaker' hardware, allowing all users to reconstruct at the same quality.
- Rendering then happens according to the device's capabilities, but the data sets are consistent across all of them!


## Important links

### Study of binary search performance on modern CPUs (especially regarding cache characteristics)

- http://www.pvk.ca/Blog/2012/07/03/binary-search-star-eliminates-star-branch-mispredictions/
- http://www.pvk.ca/Blog/2012/07/30/binary-search-is-a-pathological-case-for-caches/
