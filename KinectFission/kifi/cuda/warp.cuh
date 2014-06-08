#pragma once

#include "constants.cuh"



namespace kifi {
namespace cuda {

// TODO: Make reduce and scan work with not-power-of-2 'width's !!!

template< unsigned width = WARP_SZ >
struct warp
{
	static inline __device__ int      reduce( int      partialSum );
	static inline __device__ unsigned reduce( unsigned partialSum );
	static inline __device__ float    reduce( float    partialSum );
	static inline __device__ unsigned reduce_bit( unsigned pred );

	template< bool includeSelf > static inline __device__ int      scan( int      partialScan );
	template< bool includeSelf > static inline __device__ unsigned scan( unsigned partialScan );
	template< bool includeSelf > static inline __device__ float    scan( float    partialScan );

	template< bool includeSelf > static inline __device__ int      scan( int      partialScan, int      & outSum );
	template< bool includeSelf > static inline __device__ unsigned scan( unsigned partialScan, unsigned & outSum );
	template< bool includeSelf > static inline __device__ float    scan( float    partialScan, float    & outSum );

	template< bool includeSelf > static inline __device__ unsigned scan_bit( unsigned pred );
	template< bool includeSelf > static inline __device__ unsigned scan_bit( unsigned pred, unsigned & outSum );

private:

	static_assert( width <= WARP_SZ, "Invalid warp size" );

	template< typename T >
	static inline __device__ T _reduce( T partialSum );

	static inline __device__ int      _add_up( int      var, unsigned delta, unsigned laneIdx );
	static inline __device__ unsigned _add_up( unsigned var, unsigned delta, unsigned laneIdx );
	static inline __device__ float    _add_up( float    var, unsigned delta, unsigned laneIdx );

	template< bool includeSelf, typename T > static inline __device__ T _scan( T partialScan );
	template< bool includeSelf, typename T > static inline __device__ T _scan( T partialScan, T & outSum );
};

}} // namespace



#pragma region Implementation

#include <kifi/cuda/intrinsics.cuh>



namespace kifi {
namespace cuda {

#pragma region T reduce( T )

template< unsigned width >
__device__ int warp< width >::reduce( int partialSum )
{
	return _reduce( partialSum ); 
}
	
template< unsigned width >
__device__ unsigned warp< width >::reduce( unsigned partialSum ) 
{ 
	return _reduce( partialSum ); 
}

template< unsigned width >
__device__ float warp< width >::reduce( float partialSum ) 
{ 
	return _reduce( partialSum ); 
}

template< unsigned width >
__device__ unsigned warp< width >::reduce_bit( unsigned pred ) 
{
	return __popc( __ballot( pred ) & (~0u >> (WARP_SZ - width)) );
}

#pragma endregion

#pragma region T scan( T )

template< unsigned width >
template< bool includeSelf >
__device__ int warp< width >::scan( int partialScan )
{
	return _scan< includeSelf, int >( partialScan );
}

template< unsigned width >
template< bool includeSelf >
__device__ unsigned warp< width >::scan( unsigned partialScan )
{
	return _scan< includeSelf, unsigned >( partialScan );
}

template< unsigned width >
template< bool includeSelf >
__device__ float warp< width >::scan( float partialScan )
{
	return _scan< includeSelf, float >( partialScan );
}

#pragma endregion

#pragma region T scan( T, T & outSum )

template< unsigned width >
template< bool includeSelf >
__device__ int warp< width >::scan( int partialScan, int & outSum )
{
	return _scan< includeSelf, int >( partialScan, outSum );
}

template< unsigned width >
template< bool includeSelf >
__device__ unsigned warp< width >::scan( unsigned partialScan, unsigned & outSum )
{
	return _scan< includeSelf, unsigned >( partialScan, outSum );
}

template< unsigned width >
template< bool includeSelf >
__device__ float warp< width >::scan( float partialScan, float & outSum )
{
	return _scan< includeSelf, float >( partialScan, outSum );
}

#pragma endregion

#pragma region scan_bit

template< unsigned width >
template< bool includeSelf >
__device__ unsigned warp< width >::scan_bit( unsigned pred )
{
	unsigned result;

	if( includeSelf )
	{
		asm
		(
			"{\n\t"
			".reg .u32 lane_mask;\n\t"
			".reg .u32 warp_mask;\n\t"
			".reg .pred thread_vote;\n\t"

			"mov.u32 lane_mask, %lanemask_le;\n\t"			// inclusive
			"setp.ne.u32 thread_vote, %1, 0;\n\t"
			"vote.ballot.b32 warp_mask, thread_vote;\n\t"
			
			"and.b32 %0, warp_mask, lane_mask;\n\t"
			"popc.b32 %0, %0;\n\t"
			"}\n\t"
			:
			"=r"( result ) : "r"( pred )
		);
	}
	else
	{
		asm
		(
			"{\n\t"
			".reg .u32 lane_mask;\n\t"
			".reg .u32 warp_mask;\n\t"
			".reg .pred thread_vote;\n\t"

			"mov.u32 lane_mask, %lanemask_lt;\n\t"			// exclusive
			"setp.ne.u32 thread_vote, %1, 0;\n\t"
			"vote.ballot.b32 warp_mask, thread_vote;\n\t"
			
			"and.b32 %0, warp_mask, lane_mask;\n\t"
			"popc.b32 %0, %0;\n\t"
			"}\n\t"
			:
			"=r"( result ) : "r"( pred )
		);
	}

	return result;
}

template< unsigned width >
template< bool includeSelf >
__device__ unsigned warp< width >::scan_bit( unsigned pred, unsigned & outSum )
{
	unsigned result;
	unsigned warpMask;

	if( includeSelf )
	{
		asm
		(
			"{\n\t"
			".reg .u32 lane_mask;\n\t"
			".reg .u32 warp_mask;\n\t"
			".reg .pred thread_vote;\n\t"

			"mov.u32 lane_mask, %lanemask_le;\n\t"			// inclusive
			"setp.ne.u32 thread_vote, %2, 0;\n\t"
			"vote.ballot.b32 warp_mask, thread_vote;\n\t"
			
			"and.b32 %0, warp_mask, lane_mask;\n\t"
			"popc.b32 %0, %0;\n\t"

			"mov.u32 %1, warp_mask;\n\t"
			"}\n\t"
			:
			"=r"( result ), "=r"( warpMask ) : "r"( pred )
		);
	}
	else
	{
		asm
		(
			"{\n\t"
			".reg .u32 lane_mask;\n\t"
			".reg .u32 warp_mask;\n\t"
			".reg .pred thread_vote;\n\t"

			"mov.u32 lane_mask, %lanemask_lt;\n\t"			// exclusive
			"setp.ne.u32 thread_vote, %2, 0;\n\t"
			"vote.ballot.b32 warp_mask, thread_vote;\n\t"
			
			"and.b32 %0, warp_mask, lane_mask;\n\t"
			"popc.b32 %0, %0;\n\t"

			"mov.u32 %1, warp_mask;\n\t"
			"}\n\t"
			:
			"=r"( result ), "=r"( warpMask ) : "r"( pred )
		);
	}

	outSum = __popc( warpMask & (~0u >> (WARP_SZ - width)) );

	return result;
}

#pragma endregion

#pragma region private

template< unsigned width >
template< typename T >
__device__ T warp< width >::_reduce( T partialSum )
{
#pragma unroll
	for( int mask = width / 2; mask > 0; mask /= 2 )
		partialSum += __shfl_xor( partialSum, mask );

	return partialSum;
}



#define _kifi_add_up( type, constr_letter, var, delta, laneIdx )\
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



template< unsigned width >
__device__ int warp< width >::_add_up( int var, unsigned delta, unsigned laneIdx )
{
	_kifi_add_up( s32, r, var, delta, laneIdx )
	return var;
}

template< unsigned width >
__device__ unsigned warp< width >::_add_up( unsigned var, unsigned delta, unsigned laneIdx )
{
	_kifi_add_up( u32, r, var, delta, laneIdx )
	return var;
}

template< unsigned width >
__device__ float warp< width >::_add_up( float var, unsigned delta, unsigned laneIdx )
{
	_kifi_add_up( f32, f, var, delta, laneIdx )
	return var;
}



template< unsigned width >
template< bool includeSelf, typename T >
__device__ T warp< width >::_scan( T partialScan )
{
	unsigned const laneIdx = laneid();

	T result = partialScan;

#pragma unroll
	for( int delta = 1; delta <= width / 2; delta *= 2 )
		result = _add_up( result, delta, laneIdx );

	return result - !includeSelf * partialScan;
}

template< unsigned width >
template< bool includeSelf, typename T >
__device__ T warp< width >::_scan( T partialScan, T & outSum )
{
	T result = _scan< true, T >( partialScan );

	outSum = __shfl( result, width - 1 );

	return result - !includeSelf * partialScan;
}

#pragma endregion

}} // namespace

#pragma endregion