#pragma once

#include "constants.cuh"



namespace svcu {

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

}



#include "block.inl"