#pragma once

#include "constants.cuh"
#include "intrinsics.cuh"



#define _add_up( type, constr_letter, var, delta, laneIdx )\
\
asm\
(\
	"{\n\t"\
	".reg ."#type" tmp;\n\t"\
	".reg .pred validLane;\n\t"\
	\
	"shfl.up.b32 tmp, %0, %1, 0;\n\t"\
	"setp.ge.u32 validLane, %2, %1;\n\t"\
	"@validLane add."#type" %0, %0, tmp;\n\t"\
	"}\n\t"\
	:\
	"+"#constr_letter( var ) : "r"( delta ), "r"( laneIdx )\
);



namespace svcu {

inline __device__ int add_up( int var, unsigned const delta, unsigned const laneIdx )
{
	_add_up( s32, r, var, delta, laneIdx )
	return var;
}

inline __device__ unsigned add_up( unsigned var, unsigned const delta, unsigned const laneIdx )
{
	_add_up( u32, r, var, delta, laneIdx )
	return var;
}

inline __device__ float add_up( float var, unsigned const delta, unsigned const laneIdx )
{
	_add_up( f32, f, var, delta, laneIdx )
	return var;
}



template< typename T >
inline __device__ T warp_inclusive_scan( T partialScan, unsigned const laneIdx )
{
#pragma unroll
	for( int i = 1; i <= WARP_SZ / 2; i *= 2 )
		partialScan = add_up( partialScan,  i, laneIdx );

	return partialScan;
}



template< bool includeSelf, typename T >
inline __device__ T warp_scan( T partialScan, unsigned const laneIdx )
{
	return warp_inclusive_scan< T >( partialScan, laneIdx ) - !includeSelf * partialScan;
}

template< bool includeSelf, typename T >
inline __device__ T warp_scan( T partialScan, T & outSum, unsigned const laneIdx )
{
	T result = warp_inclusive_scan< T >( partialScan, laneIdx );

	outSum = __shfl( result, WARP_SZ - 1 );

	return result - !includeSelf * partialScan;
}



template< int NT, bool includeSelf, typename T >
inline __device__ T block_scan
(
	T partialScan, T * shared, 
	unsigned const laneIdx, unsigned const warpIdx 
)
{
	T warpSum;
	partialScan = warp_scan< includeSelf, T >( partialScan, warpSum, laneIdx );

	if( 0 == laneIdx )
		shared[ warpIdx ] = warpSum;
	__syncthreads();

	T offset = 0;
#pragma unroll
	for( int i = 0; i < NT / WARP_SZ; i++ )
		offset += shared[ i ] * (i < warpIdx); // '<' means exclusive

	return partialScan + offset;
}

template< int NT, bool includeSelf, typename T >
inline __device__ T block_scan
(
	T partialScan, T & outSum, T * shared, 
	unsigned const laneIdx, unsigned const warpIdx 
)
{
	T warpSum;
	partialScan = warp_scan< includeSelf, T >( partialScan, warpSum, laneIdx );

	if( 0 == laneIdx )
		shared[ warpIdx ] = warpSum;
	__syncthreads();

	outSum = 0;
	T offset = 0;
#pragma unroll
	for( int i = 0; i < NT / WARP_SZ; i++ )
	{
		T tmp = shared[ i ];
		outSum += tmp;
		offset += tmp * (i < warpIdx); // '<' means exclusive
	}

	return partialScan + offset;
}



template< int NT, bool includeSelf, typename T >
inline __device__ T block_scan( T partialScan, T * shared )
{
	return block_scan< NT, includeSelf, T >
	(
		partialScan, shared, 
		threadIdx.x % WARP_SZ, threadIdx.x / WARP_SZ
	);
}

template< int NT, bool includeSelf, typename T >
inline __device__ T block_scan( T partialScan, T & outSum, T * shared )
{
	return block_scan< NT, includeSelf, T >
	(
		partialScan, outSum, shared, 
		threadIdx.x % WARP_SZ, threadIdx.x / WARP_SZ
	);
}

}