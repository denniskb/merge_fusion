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
	__shared__ unsigned shared[ NT ];

	for( unsigned bid = blockIdx.x, end = size / NT; bid < end; bid += gridDim.x )
	{
		//int x = svcu::block_reduce< NT, false >( bid, shared );
		//int x = svcu::block_reduce( (bool)bid );
		unsigned x = svcu::warp_scan< false >( (bool)bid );
		//int x = svcu::warp_reduce( (bool)bid );

		//svcu::block_scan< NT, true >( (unsigned)bid, shared );

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