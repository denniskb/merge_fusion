#include "reduce.h"

#include <cassert>

#include <vector_types.h>

#include "constants.cuh"
#include "cuda_device.h"
#include "helper_math_ext.h"
#include "reduce.cuh"



template< unsigned NT, unsigned SEG_SZ >
static __global__ void _segmented_reduce( unsigned const * data, unsigned size, unsigned * outSums )
{	
	unsigned const NW = NT / WARP_SZ;
	unsigned const VT = SEG_SZ / WARP_SZ;

	unsigned const warpIdx = warpid();
	unsigned const laneIdx = laneid();

	for
	(
		int wid = warpIdx + blockIdx.x * NW,
		end = size / SEG_SZ,
		step = gridDim.x * NW;
		wid < end; 
		wid += step 
	)
	{		
		unsigned sum = 0;

#pragma unroll
		for
		(
			int i = 0, 
			src = laneIdx + wid * ( SEG_SZ / 4 ); 
			i < VT / 4; 
			i++, src += WARP_SZ 
		)
			sum += horizontal_sum( reinterpret_cast< uint4 const * >( data )[ src ] );

		sum = svcu::warp_reduce( sum );

		__syncthreads();
		if( 0 == laneIdx )
			outSums[ wid ] = sum;
	}

	if( 0 == blockIdx.x && 0 == warpIdx && size % SEG_SZ )
	{
		unsigned sum = 0;
		for( int tid = laneIdx + size / SEG_SZ * SEG_SZ; tid < size; tid += WARP_SZ )
			sum += data[ tid ];

		sum = svcu::warp_reduce( sum );

		if( 0 == laneIdx )
			outSums[ size / SEG_SZ ] = sum;
	}
}



void svcu::segmented_reduce
(
	unsigned const * data, unsigned size,
	unsigned segmentSize,

	unsigned * outSums
)
{
	int nBlocks = cuda_device::main().max_residing_blocks( 128, 32, 0 );

	switch( segmentSize )
	{
		case  128: _segmented_reduce< 128,  128 ><<< nBlocks, 128 >>>( data, size, outSums ); break;
		case  256: _segmented_reduce< 128,  256 ><<< nBlocks, 128 >>>( data, size, outSums ); break;
		case  512: _segmented_reduce< 128,  512 ><<< nBlocks, 128 >>>( data, size, outSums ); break;
		case 1024: _segmented_reduce< 128, 1024 ><<< nBlocks, 128 >>>( data, size, outSums ); break;
		default: assert( false );
	}
}