#pragma once

#ifdef __CUDACC__
#define kppl_host __host__
#define kppl_device __device__
#define kppl_both __host__ __device__
#else
#define kppl_host
#define kppl_device
#define kppl_both
#endif

#ifdef __CUDA_ARCH__
#define kppl_assert( X )
#else
#define kppl_assert( X ) assert( X )
#endif