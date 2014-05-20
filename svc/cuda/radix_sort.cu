#include "radix_sort.h"

#include "helper_math_ext.h"
#include "scan.cuh"



static __global__ void _radix_sort
(
	unsigned * data, unsigned size,
	unsigned * tmp
)
{
	__shared__ int shared[ 8 ];

	unsigned const laneIdx = threadIdx.x % WARP_SZ;
	unsigned const warpIdx = threadIdx.x / WARP_SZ;

	for( int bid = blockIdx.x, end = size / 256; bid < end; bid += gridDim.x )
	{
		int x = svcu::block_scan< 256, true >( bid, shared, laneIdx, warpIdx );

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
	_radix_sort<<< 256, 48 >>>( data, size, tmp );
}