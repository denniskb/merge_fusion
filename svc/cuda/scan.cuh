#pragma once

#include "constants.h"



namespace svcu {

template< typename T >
inline __device__ T shfl( T var, int srcLane )
{
	asm
	(
		"{\n\t"
		"shfl.idx.b32 %0, %0, %1, 0x1f;\n\t"
		"}"
		:
		"+r"( var ) : "r"( srcLane )
	);

	return var;
}



inline __device__ int add_up( int var, int delta )
{
	asm
	(
		"{\n\t"
		".reg .s32 tmp;\n\t"
		".reg .pred validLane;\n\t"

		"shfl.up.b32 tmp|validLane, %0,  %1, 0;\n\t"
		"@validLane add.s32 %0, %0, tmp;\n\t"
		"}"
		:
		"+r"( var ) : "r"( delta )
	);

	return var;
}

inline __device__ unsigned add_up( unsigned var, int delta )
{
	asm
	(
		"{\n\t"
		".reg .u32 tmp;\n\t"
		".reg .pred validLane;\n\t"

		"shfl.up.b32 tmp|validLane, %0,  %1, 0;\n\t"
		"@validLane add.u32 %0, %0, tmp;\n\t"
		"}"
		:
		"+r"( var ) : "r"( delta )
	);

	return var;
}

inline __device__ float add_up( float var, int delta )
{
	asm
	(
		"{\n\t"
		".reg .f32 tmp;\n\t"
		".reg .pred validLane;\n\t"

		"shfl.up.b32 tmp|validLane, %0,  %1, 0;\n\t"
		"@validLane add.f32 %0, %0, tmp;\n\t"
		"}"
		:
		"+f"( var ) : "r"( delta )
	);

	return var;
}



template< typename T >
inline __device__ T warp_inclusive_scan( T partialScan )
{
#pragma unroll
	for( int i = 1; i <= 16; i *= 2 )
		partialScan = add_up( partialScan,  i );

	return partialScan;
}

template< typename T >
inline __device__ T warp_inclusive_scan( T partialScan, T & outSum )
{
	T result = warp_inclusive_scan( partialScan );

	outSum = shfl< T >( result, WARP_SZ - 1 );

	return result;
}



template< typename T >
inline __device__ T warp_exclusive_scan( T partialScan )
{
	T tmp = partialScan;

	partialScan = warp_inclusive_scan( partialScan );

	return partialScan - tmp;
}

template< typename T >
inline __device__ T warp_exclusive_scan( T partialScan, T & outSum )
{
	T tmp = partialScan;

	partialScan = warp_inclusive_scan( partialScan, outSum );

	return partialScan - tmp;
}



template< bool includeSelf, typename T >
inline __device__ T warp_scan( T partialScan )
{
	if( includeSelf )
		return warp_inclusive_scan( partialScan );
	else
		return warp_exclusive_scan( partialScan );
}

template< bool includeSelf, typename T >
inline __device__ T warp_scan( T partialScan, T & outSum )
{
	if( includeSelf )
		return warp_inclusive_scan( partialScan, outSum );
	else
		return warp_exclusive_scan( partialScan, outSum );
}

}