#pragma once

#ifdef __CUDACC__
#define svc_host __host__
#define svc_device __device__
#define svc_both __host__ __device__
#else
#define svc_host
#define svc_device
#define svc_both
#endif

#ifdef __CUDA_ARCH__
#define svc_assert( X )
#else
#define svc_assert( X ) assert( X )
#endif