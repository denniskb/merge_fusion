#pragma once

#include "constants.cuh"



namespace svcu {

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

}



#include "warp.inl"