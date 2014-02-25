#pragma once

#include "force_nvcc.h"

#include <vector_functions.h>
#include <vector_types.h>

#include "HostVolume.h"
#include "Voxel.m"



namespace kppl {

class KernelVolume
{
public:
	inline __host__ KernelVolume( DeviceVolume & copy ) :
		data( copy.Data() ),
		res( copy.Resolution() ),
		truncMargin( copy.TruncationMargin() ),

		vLen( copy.SideLength() / res ),
		m_resOver2MinusPoint5TimesVoxelLenNeg( -( res / 2 - 0.5f ) * ( copy.SideLength() / res ) )
	{
	}

	inline __device__ int Slice() const
	{
		return res * res;
	}

	inline __device__ int Volume() const
	{
		return res * Slice();
	}



	inline __device__ float4 VoxelCenter( int x, int y, int z ) const
	{
		return make_float4
		( 
			x * vLen + m_resOver2MinusPoint5TimesVoxelLenNeg, 
			y * vLen + m_resOver2MinusPoint5TimesVoxelLenNeg, 
			z * vLen + m_resOver2MinusPoint5TimesVoxelLenNeg, 
			1.0f 
		);
	}



	inline __device__ Voxel operator()( int x, int y, int z ) const
	{
		return data[ Index3Dto1D( x, y, z, res ) ];
	}



	inline static __device__ int Index3Dto1D( unsigned x, unsigned y, unsigned z, unsigned res )
	{
		return ( z * res + y ) * res + x;
	}

	Voxel * data;
	int const res;
	float const truncMargin;

	// cached values..
	float const vLen;

private:
	// ..cached values
	float const m_resOver2MinusPoint5TimesVoxelLenNeg;
};

}