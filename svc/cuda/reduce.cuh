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

}