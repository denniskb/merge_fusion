#pragma once

#include "constants.cuh"
#include "intrinsics.cuh"



namespace svcu {

template< typename T >
inline __device__ T warp_reduce( T partialSum )
{
#pragma unroll
	for( int mask = WARP_SZ / 2; mask > 0; mask /= 2 )
		partialSum += __shfl_xor( partialSum, mask );

	return partialSum;
}



template< int NT, bool broadCastSum, typename T >
inline __device__ T block_reduce
(
	T partialSum, T * shared
)
{
	shared[ threadIdx.x ] = partialSum;
	__syncthreads();

	T totalSum = 0;

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

}