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
	unsigned const laneIdx 
)
{
	int const VT = NT / WARP_SZ;

	shared[ threadIdx.x ] = partialScan;
	__syncthreads();

	if( threadIdx.x < WARP_SZ )
	{
		T data[ VT ];

		T offset = T();
#pragma unroll
		for( int i = 0; i < VT; i++ )
		{
			if( includeSelf )
				offset += shared[ VT * threadIdx.x + i ];

			data[ i ] = offset;

			if ( ! includeSelf )
				offset += shared[ VT * threadIdx.x + i ];
		}

		offset = warp_scan< false >( offset, laneIdx );

		for( int i = 0; i < VT; i++ )
			shared[ VT * threadIdx.x + i ] = data[ i ] + offset;
	}
	__syncthreads();
}

template< int NT, bool includeSelf, typename T >
inline __device__ T block_scan
(
	T partialScan, T & shOutSum, T * shared, 
	unsigned const laneIdx 
)
{
	int const VT = NT / WARP_SZ;

	shared[ threadIdx.x ] = partialScan;
	__syncthreads();

	if( threadIdx.x < WARP_SZ )
	{
		T data[ VT ];

		T offset = T();
#pragma unroll
		for( int i = 0; i < VT; i++ )
		{
			if( includeSelf )
				offset += shared[ VT * threadIdx.x + i ];

			data[ i ] = offset;

			if ( ! includeSelf )
				offset += shared[ VT * threadIdx.x + i ];
		}

		T sum;
		offset = warp_scan< false >( offset, sum, laneIdx );

		for( int i = 0; i < VT; i++ )
			shared[ VT * threadIdx.x + i ] = data[ i ] + offset;

		if( 0 == threadIdx.x )
			shOutSum = sum;
	}
	__syncthreads();
}



template< int NT, bool includeSelf, typename T >
inline __device__ T block_scan( T partialScan, T * shared )
{
	return block_scan< NT, includeSelf, T >
	(
		partialScan, shared, 
		threadIdx.x % WARP_SZ
	);
}

template< int NT, bool includeSelf, typename T >
inline __device__ T block_scan( T partialScan, T & shOutSum, T * shared )
{
	return block_scan< NT, includeSelf, T >
	(
		partialScan, shOutSum, shared, 
		threadIdx.x % WARP_SZ
	);
}

}