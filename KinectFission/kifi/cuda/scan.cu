#include <cassert>

#include <vector_types.h>

#include <kifi/cuda/constants.cuh>
#include <kifi/cuda/cuda_device.h>
#include <kifi/cuda/helper_math_ext.h>
#include <kifi/cuda/scan.h>
#include <kifi/cuda/warp.cuh>

using namespace kifi::cuda;



template< unsigned NT, unsigned SEG_SZ, bool includeSelf >
static __global__ void _segmented_scan
(
	unsigned const * data, unsigned const size, 

	unsigned * out
)
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
		unsigned segmentOffset = 0;

#pragma unroll
		for
		(
			int i = 0, 
			tid = laneIdx + wid * ( SEG_SZ / 4 ); 
			i < VT / 4; 
			i++, tid += WARP_SZ 
		)
		{
			uint4 word = reinterpret_cast< uint4 const * >( data )[ tid ];

			unsigned warpOffset, warpSum;
			if( i < VT / 4 - 1 )
				warpOffset = warp<>::scan< false >( horizontal_sum( word ), warpSum );
			else
				warpOffset = warp<>::scan< false >( horizontal_sum( word ) );
			
			warpOffset += segmentOffset;
			segmentOffset += warpSum;

			scan< includeSelf >( word );
			word += warpOffset;

			reinterpret_cast< uint4 * >( out )[ tid ] = word;
		}
		__syncthreads();
	}

	if( 0 == blockIdx.x && 0 == warpIdx && size % SEG_SZ )
	{
		unsigned segmentOffset = 0;

#pragma unroll
		for
		(
			int i = 0,
			tid = size / SEG_SZ * SEG_SZ + laneIdx;
			i < VT;
			i++, tid += WARP_SZ )
		{
			unsigned word = tid < size ? data[ tid ] : 0;

			unsigned warpSum;
			word = warp<>::scan< includeSelf >( word, warpSum );

			word += segmentOffset;
			segmentOffset += warpSum;

			if( tid < size )
				out[ tid ] = word;
		}
	}
}



template< bool includeSelf >
static void segmented_scan
(
	unsigned const * data, unsigned size,
	unsigned segmentSize,

	unsigned * out
)
{
	int nBlocks = cuda_device::main().max_residing_blocks( 128, 32, 0 );

	switch( segmentSize )
	{
		case  128: _segmented_scan< 128,  128, includeSelf ><<< nBlocks, 128 >>>( data, size, out ); break;
		case  256: _segmented_scan< 128,  256, includeSelf ><<< nBlocks, 128 >>>( data, size, out ); break;
		case  512: _segmented_scan< 128,  512, includeSelf ><<< nBlocks, 128 >>>( data, size, out ); break;
		case 1024: _segmented_scan< 128, 1024, includeSelf ><<< nBlocks, 128 >>>( data, size, out ); break;
		default: assert( false );
	}
}



namespace kifi {
namespace cuda {

void segmented_inclusive_scan
(
	unsigned const * data, unsigned size,
	unsigned segmentSize,

	unsigned * out
)
{
	segmented_scan< true >( data, size, segmentSize, out );
}

void segmented_exclusive_scan
(
	unsigned const * data, unsigned size,
	unsigned segmentSize,

	unsigned * out
)
{
	segmented_scan< false >( data, size, segmentSize, out );
}

}} // namespace