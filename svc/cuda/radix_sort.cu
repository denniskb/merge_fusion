#include "radix_sort.h"

#include <cstdio>

#include <vector_types.h>

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
	__shared__ unsigned stage[ NT ];

	unsigned const laneIdx = laneid();
	unsigned const warpIdx = warpid();

	for( int i = blockIdx.x, end = size / NT; i < end; i += gridDim.x )
	{
		unsigned tid = threadIdx.x + i * NT;

		unsigned word = data[ tid ];

		for( int j = 0; j < 4; j++ )
		{
			unsigned bits = ( word >> j ) & 0x1;

			unsigned no1sTotal;
			unsigned no1sBeforeMe = block< NT >::scan_bit< false >( bits, no1sTotal, shared );

			unsigned no0sTotal = NT - no1sTotal;
			unsigned no0sBeforeMe = threadIdx.x - no1sBeforeMe;

			unsigned offset = bits * (no0sTotal + no1sBeforeMe) + (1 - bits) * no0sBeforeMe;

			stage[ offset ] = word;
			__syncthreads();

			word = stage[ threadIdx.x ];
		}

		tmp[ tid ] = word;
	}
}



void svcu::radix_sort
(
	unsigned * data, unsigned size, 
	unsigned * tmp 
)
{
	unsigned const NT = 128;

	//_radix_sort< NT ><<< (96 / (NT / 128)), NT >>>( data, size, tmp );
	_radix_sort< 256 ><<< 48, 256 >>>( data, size, tmp );
}