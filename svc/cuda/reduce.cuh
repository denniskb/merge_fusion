#pragma once

#include "constants.h"



namespace svcu {

template< typename T >
inline __device__ T shfl_xor( T var, int laneMask )
{
	asm
	(
		"{\n\t"
		"shfl.bfly.b32 %0, %0, %1, 0x1f;\n\t"
		"}"
		:
		"+r"( var ) : "r"( laneMask )
	);

	return var;
}



template< typename T >
inline __device__ T warp_reduce( T partialSum )
{
#pragma unroll
	for( int mask = WARP_SZ / 2; mask > 0; mask /= 2 )
		partialSum += shfl_xor< T >( partialSum, mask );

	return partialSum;
}



template< int NT, typename T >
inline __device__ T block_reduce
(
	T partialSum, T * shared, 
	unsigned const laneIdx, unsigned const warpIdx 
)
{
	partialSum = warp_reduce< T >( partialSum );

	if( 0 == laneIdx )
		shared[ warpIdx ] = partialSum;
	__syncthreads();

	T totalSum = 0;
#pragma unroll
	for( int i = 0; i < NT / WARP_SZ; i++ )
		totalSum += shared[ i ];

	return totalSum;
}

template< int NT, typename T >
inline __device__ T block_reduce( T partialSum, T * shared )
{
	return block_reduce< NT, T >( partialSum, shared, threadIdx.x % WARP_SZ, threadIdx.x / WARP_SZ );
}

}