#include "radix_sort.h"

#include "helper_math_ext.h"
#include "reduce.cuh"
#include "scan.cuh"



template< unsigned NT >
static __global__ void _radix_sort
(
	unsigned * data, unsigned size,
	unsigned * tmp
)
{
	__shared__ int shared[ NT ];

	unsigned const laneIdx = threadIdx.x % 32;
	unsigned const warpIdx = threadIdx.x / 32;

	for( int bid = blockIdx.x, end = size / (NT * 4); bid < end; bid += gridDim.x )
	{
		//int x = svcu::block_reduce< NT, false >( bid, shared );
		//int x = svcu::block_scan< NT, true >( bid, shared );
		int y;
		svcu::warp_scan< true >( 1, y, laneIdx );
		//int x = svcu::warp_reduce( bid );

		tmp[ threadIdx.x + bid * NT ] = y;
	}
}



void svcu::radix_sort
(
	unsigned * data, unsigned size, 
	unsigned * tmp 
)
{
	unsigned const NT = 128;

	_radix_sort< NT ><<< (96 / (NT / 128)), NT >>>( data, size, tmp );
}