#pragma once

#include "constants.cuh"
#include "intrinsics.cuh"



namespace svcu {

template< typename T >
inline __device__ T _warp_reduce( T partialSum )
{
#pragma unroll
	for( int mask = WARP_SZ / 2; mask > 0; mask /= 2 )
		partialSum += __shfl_xor( partialSum, mask );

	return partialSum;
}

inline __device__ int warp_reduce( int partialSum ) { return _warp_reduce( partialSum ); }
inline __device__ unsigned warp_reduce( unsigned partialSum ) { return _warp_reduce( partialSum ); }
inline __device__ float warp_reduce( float partialSum ) { return _warp_reduce( partialSum ); }

inline __device__ unsigned warp_reduce( bool partialSum ) 
{
	return __popc( __ballot( partialSum ) );
}



template< int NT, bool broadCastSum, typename T >
inline __device__ T _block_reduce
(
	T partialSum, T * shared
)
{
	shared[ threadIdx.x ] = partialSum;
	__syncthreads();

	T totalSum = T();

	if( threadIdx.x < WARP_SZ )
	{
#pragma unroll
		for( int i = 0; i < (NT / WARP_SZ); i++ )
			totalSum += shared[ threadIdx.x + i * WARP_SZ ];

		totalSum = warp_reduce( totalSum );
		
		if( broadCastSum )
			shared[ threadIdx.x ] = totalSum;
	}

	if( broadCastSum )
		__syncthreads();

	return totalSum;
}

template< int NT, bool broadCastSum >
inline __device__ int block_reduce( int partialSum, int * shared )
{
	return _block_reduce< NT, broadCastSum, int >( partialSum, shared );
}

template< int NT, bool broadCastSum >
inline __device__ unsigned block_reduce( unsigned partialSum, unsigned * shared )
{
	return _block_reduce< NT, broadCastSum, unsigned >( partialSum, shared );
}

template< int NT, bool broadCastSum >
inline __device__ float block_reduce( float partialSum, float * shared )
{
	return _block_reduce< NT, broadCastSum, float >( partialSum, shared );
}

inline __device__ unsigned block_reduce( bool partialSum )
{
	return __syncthreads_count( partialSum );
}

}