#include "radix_sort.h"

#include "helper_math_ext.h"
#include "reduce.cuh"



static __global__ void _radix_sort
(
	unsigned * data, unsigned size,
	unsigned * tmp
)
{
	__shared__ int ping[ 256 ];

	unsigned const laneIdx = threadIdx.x % WARP_SZ;
	unsigned const warpIdx = threadIdx.x / WARP_SZ;

	for( int bid = blockIdx.x, end = size / 512; bid < end; bid += gridDim.x )
	{
		int x = svcu::warp_reduce( bid );

		if( threadIdx.x == 0 )
			tmp[ bid ] = x;
	}
}



void svcu::radix_sort
(
	unsigned * data, unsigned size, 
	unsigned * tmp 
)
{
	_radix_sort<<< 96, 128 >>>( data, size, tmp );
}