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
		sideLen( copy.SideLength() ),
		truncMargin( copy.TruncationMargin() ),

		packScale( 2047.5f / truncMargin ),
		unpackScale( truncMargin / 2047.5f ),
		m_vLen( sideLen / res ),
		m_resOver2MinusPoint5TimesVoxelLenNeg( -( res / 2 - 0.5f ) * ( sideLen / res ) )
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
			x * m_vLen + m_resOver2MinusPoint5TimesVoxelLenNeg, 
			y * m_vLen + m_resOver2MinusPoint5TimesVoxelLenNeg, 
			z * m_vLen + m_resOver2MinusPoint5TimesVoxelLenNeg, 
			1.0f 
		);
	}

	inline __device__ Voxel & operator()( int x, int y, int z )
	{
		return data[ Index3Dto1D( x, y, z, res ) ];
	}

	Voxel * data;
	int const res;
	float const sideLen;
	float const truncMargin;

	// cached values..
	float const packScale;
	float const unpackScale;

private:
	// ..cached values
	float const m_vLen;
	float const m_resOver2MinusPoint5TimesVoxelLenNeg;

	inline static __device__ int Index3Dto1D( int x, int y, int z, int res )
	{
		return ( z * res + y ) * res + x;
	}
};

}