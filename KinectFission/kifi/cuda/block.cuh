#pragma once

#include <kifi/cuda/constants.cuh>



namespace kifi {
namespace cuda {

template< unsigned NT >
struct block
{
	static inline __device__ int      reduce( int      partialSum, int      * shared );
	static inline __device__ unsigned reduce( unsigned partialSum, unsigned * shared );
	static inline __device__ float    reduce( float    partialSum, float    * shared );
	static inline __device__ unsigned reduce_bit( unsigned pred );

	template< bool includeSelf, typename T > static inline __device__ void scan( T partialScan, T * shared );
	template< bool includeSelf, typename T > static inline __device__ void scan( T partialScan, T & shOutSum, T * shared );

	template< bool includeSelf > static inline __device__ unsigned scan_bit( unsigned pred, unsigned * shared );
	template< bool includeSelf > static inline __device__ unsigned scan_bit( unsigned pred, unsigned & outSum, unsigned * shared );

private:

	static_assert( NT % WARP_SZ == 0, "Invalid block size" );

	template< typename T > static inline __device__ T _reduce( T partialSum, T * shared );
};

}} // namespace



#pragma region Implementation

#include <kifi/cuda/intrinsics.cuh>
#include <kifi/cuda/warp.cuh>



namespace kifi {
namespace cuda {

#pragma region reduce

template< unsigned NT >
__device__ int block< NT >::reduce( int partialSum, int * shared )
{
	return _reduce< int >( partialSum, shared );
}

template< unsigned NT >
__device__ unsigned block< NT >::reduce( unsigned partialSum, unsigned * shared )
{
	return _reduce< unsigned >( partialSum, shared );
}

template< unsigned NT >
__device__ float block< NT >::reduce( float partialSum, float * shared )
{
	return _reduce< float >( partialSum, shared );
}

template< unsigned NT >
__device__ unsigned block< NT >::reduce_bit( unsigned pred )
{
	return __syncthreads_count( pred );
}

#pragma endregion

#pragma region scan

template< unsigned NT >
template< bool includeSelf, typename T >
__device__ void block< NT >::scan( T partialScan, T * shared )
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

		offset = warp<>::scan< false >( offset );

		for( int i = 0; i < VT; i++ )
			shared[ VT * threadIdx.x + i ] = data[ i ] + offset;
	}
	__syncthreads();
}

template< unsigned NT >
template< bool includeSelf, typename T >
__device__ void block< NT >::scan( T partialScan, T & shOutSum, T * shared )
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
		offset = warp<>::scan< false >( offset, sum );

		for( int i = 0; i < VT; i++ )
			shared[ VT * threadIdx.x + i ] = data[ i ] + offset;

		if( 0 == threadIdx.x )
			shOutSum = sum;
	}
	__syncthreads();
}

#pragma endregion

#pragma region scan_bit

template< unsigned NT >
template< bool includeSelf >
__device__ unsigned block< NT >::scan_bit( unsigned pred, unsigned * shared )
{
	unsigned const warpIdx = warpid();

	unsigned warpSum;
	pred = warp<>::scan_bit< includeSelf >( pred, warpSum );

	if( 0 == laneid() )
		shared[ warpIdx ] = warpSum;
	__syncthreads();

	if( threadIdx.x < WARP_SZ )
	{
		unsigned offset = shared[ threadIdx.x ] ;		
		offset = warp< NT / WARP_SZ >::scan< false >( offset );		
		shared[ threadIdx.x ] = offset;
	}
	__syncthreads();

	return pred + shared[ warpIdx ];
}

template<>
template< bool includeSelf >
__device__ unsigned block< 128 >::scan_bit( unsigned pred, unsigned * shared )
{
	unsigned const warpIdx = warpid();

	unsigned warpSum;
	pred = warp<>::scan_bit< includeSelf >( pred, warpSum );

	if( 0 == laneid() )
		shared[ warpIdx ] = warpSum;
	__syncthreads();

	unsigned offset = 0;

#pragma unroll
	for( int i = 0; i < NT / WARP_SZ; i++ )
		offset += shared[ i ] * (i < warpIdx);

	return pred + offset;
}



template< unsigned NT >
template< bool includeSelf >
__device__ unsigned block< NT >::scan_bit( unsigned pred, unsigned & outSum, unsigned * shared )
{
	unsigned const warpIdx = warpid();

	unsigned warpSum;
	unsigned warpOffset = warp<>::scan_bit< includeSelf >( pred, warpSum );

	if( 0 == laneid() )
		shared[ warpIdx ] = warpSum;
	outSum = __syncthreads_count( pred );

	if( threadIdx.x < WARP_SZ )
	{
		unsigned blockOffset = shared[ threadIdx.x ] ;		
		blockOffset = warp< NT / WARP_SZ >::scan< false >( blockOffset );		
		shared[ threadIdx.x ] = blockOffset;
	}
	__syncthreads();

	return warpOffset + shared[ warpIdx ];
}

template<>
template< bool includeSelf >
__device__ unsigned block< 128 >::scan_bit( unsigned pred, unsigned & outSum, unsigned * shared )
{
	unsigned const warpIdx = warpid();

	unsigned warpSum;
	unsigned warpOffset = warp<>::scan_bit< includeSelf >( pred, warpSum );

	if( 0 == laneid() )
		shared[ warpIdx ] = warpSum;
	outSum = __syncthreads_count( pred );

	unsigned blockOffset = 0;

#pragma unroll
	for( int i = 0; i < NT / WARP_SZ; i++ )
		blockOffset += shared[ i ] * (i < warpIdx);

	return warpOffset + blockOffset;
}

#pragma endregion

#pragma region private

template< unsigned NT >
template< typename T >
__device__ T block< NT >::_reduce( T partialSum, T * shared )
{
	shared[ threadIdx.x ] = partialSum;
	__syncthreads();

	T totalSum = T();

	if( threadIdx.x < WARP_SZ )
	{
	#pragma unroll
		for( int i = 0; i < (NT / WARP_SZ); i++ )
			totalSum += shared[ threadIdx.x + i * WARP_SZ ];

		totalSum = warp<>::reduce( totalSum );
	}

	return totalSum;
}

#pragma endregion

}} // namespace

#pragma endregion