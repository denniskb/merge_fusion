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

inline __device__ int add_up( int var, unsigned delta, unsigned laneIdx )
{
	_add_up( s32, r, var, delta, laneIdx )
	return var;
}

inline __device__ unsigned add_up( unsigned var, unsigned delta, unsigned laneIdx )
{
	_add_up( u32, r, var, delta, laneIdx )
	return var;
}

inline __device__ float add_up( float var, unsigned delta, unsigned laneIdx )
{
	_add_up( f32, f, var, delta, laneIdx )
	return var;
}



template< bool includeSelf, typename T >
inline __device__ T _warp_scan( T partialScan )
{
	unsigned const laneIdx = laneid();

	T result = partialScan;

#pragma unroll
	for( int i = 1; i <= WARP_SZ / 2; i *= 2 )
		result = add_up( result,  i, laneIdx );

	return result - !includeSelf * partialScan;
}

template< bool includeSelf >
inline __device__ int warp_scan( int partialScan )
{
	return _warp_scan< includeSelf, int >( partialScan );
}

template< bool includeSelf >
inline __device__ unsigned warp_scan( unsigned partialScan )
{
	return _warp_scan< includeSelf, unsigned >( partialScan );
}

template< bool includeSelf >
inline __device__ float warp_scan( float partialScan )
{
	return _warp_scan< includeSelf, float >( partialScan );
}

template< bool includeSelf >
inline __device__ unsigned warp_scan( bool partialScan )
{
	unsigned result;

	if( includeSelf )
	{
		asm
		(
			"{\n\t"
			".reg .u32 warp_mask;\n\t"
			".reg .u32 lane_mask;\n\t"
			".reg .pred thread_vote;\n\t"

			"setp.ne.u32 thread_vote, %1, 0;\n\t"
			"vote.ballot.b32 warp_mask, thread_vote;\n\t"
			"mov.u32 lane_mask, %lanemask_le;\n\t"			// inclusive
			
			"and.b32 %0, warp_mask, lane_mask;\n\t"
			"popc.b32 %0, %0;\n\t"
			"}\n\t"
			:
			"=r"( result ) : "r"( (unsigned) partialScan )
		);
	}
	else
	{
		asm
		(
			"{\n\t"
			".reg .u32 warp_mask;\n\t"
			".reg .u32 lane_mask;\n\t"
			".reg .pred thread_vote;\n\t"

			"setp.ne.u32 thread_vote, %1, 0;\n\t"
			"vote.ballot.b32 warp_mask, thread_vote;\n\t"
			"mov.u32 lane_mask, %lanemask_lt;\n\t"			// exclusive
			
			"and.b32 %0, warp_mask, lane_mask;\n\t"
			"popc.b32 %0, %0;\n\t"
			"}\n\t"
			:
			"=r"( result ) : "r"( (unsigned) partialScan )
		);
	}

	return result;
}

template< int NT, bool includeSelf, typename T >
inline __device__ void block_scan
(
	T partialScan, T * shared
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
			if( includeSelf )	offset += shared[ VT * threadIdx.x + i ];
			data[ i ] = offset;
			if( ! includeSelf ) offset += shared[ VT * threadIdx.x + i ];
		}

		offset = warp_scan< false >( offset );

		for( int i = 0; i < VT; i++ )
			shared[ VT * threadIdx.x + i ] = data[ i ] + offset;
	}
	__syncthreads();
}



template< bool includeSelf, typename T >
inline __device__ T warp_scan( T partialScan, T & outSum )
{
	T result = _warp_scan< true, T >( partialScan );

	outSum = __shfl( result, WARP_SZ - 1 );

	return result - !includeSelf * partialScan;
}

template< int NT, bool includeSelf, typename T >
inline __device__ void block_scan
(
	T partialScan, T & shOutSum, T * shared
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
			if( includeSelf )	offset += shared[ VT * threadIdx.x + i ];
			data[ i ] = offset;
			if( ! includeSelf ) offset += shared[ VT * threadIdx.x + i ];
		}

		T sum;
		offset = warp_scan< false >( offset, sum );

		for( int i = 0; i < VT; i++ )
			shared[ VT * threadIdx.x + i ] = data[ i ] + offset;

		if( 0 == threadIdx.x )
			shOutSum = sum;
	}
	__syncthreads();
}

}