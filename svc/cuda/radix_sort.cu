#include "radix_sort.h"

#include "block.cuh"
#include "helper_math_ext.h"
#include "warp.cuh"

using namespace svcu;



template< unsigned NT >
static __global__ void _radix_sort
(
	unsigned * data, unsigned size,
	unsigned * tmp
)
{
	__shared__ unsigned shared[ NT ];

	unsigned const laneIdx = laneid();
	unsigned const warpIdx = warpid();

	for( unsigned bid = blockIdx.x, end = size / NT; bid < end; bid += gridDim.x )
	{
		//unsigned x = warp<>::reduce( bid );
		//unsigned x = warp<>::reduce_bit( bid );
		//unsigned x = warp<>::scan< true >( bid );
		//unsigned x = warp<>::scan_bit< true >( bid );
		//unsigned x; unsigned y = warp<>::scan< true >( bid, x );
		//unsigned x; unsigned y = warp<>::scan_bit< true >( bid, x );

		//unsigned x = block< NT >::reduce( bid, shared );
		//unsigned x = block< NT >::reduce_bit( bid );

		//block< NT >::scan< true >( bid, shared ); unsigned x = shared[ threadIdx.x ];
		unsigned x = block< NT >::scan_bit< true >( bid, shared );
		
		//unsigned x; block< NT >::scan< true >( bid, x, shared );
		//unsigned x; block< NT >::scan_bit< true >( bid, x, shared );
		
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
	unsigned const NT = 128;

	_radix_sort< NT ><<< (96 / (NT / 128)), NT >>>( data, size, tmp );
}