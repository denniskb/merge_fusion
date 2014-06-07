#pragma once

#include <helper_math.h>
#include <host_defines.h>



namespace kifi {
namespace cuda {

class Voxel
{
public:
	inline __device__ Voxel( unsigned data = 0 ) :
		m_data( 0 )
	{
	}

	inline __device__ operator unsigned() const
	{
		return m_data;
	}



	inline __device__ float Distance( float truncationMargin ) const
	{
		return UnpackDistance( m_data >> 16, truncationMargin );
	}

	inline __device__ unsigned Weight() const
	{
		return m_data & 0xff;
	}



	inline __device__ void Update( float distance, float truncationMargin, unsigned weight = 1 )
	{
		unsigned newWeight = Weight() + weight;
		float newDistance = lerp
		( 
			Distance( truncationMargin ), 
			clamp( distance, -truncationMargin, truncationMargin ), 
			(float)weight / newWeight
		);

		m_data = PackDistance( newDistance, truncationMargin ) << 16 | newWeight;
	}

private:
	unsigned m_data;

	inline static __device__ unsigned PackDistance( float distance, float truncationMargin )
	{
		int d = (int) fmaf( distance / truncationMargin, 32767.5f, 32767.5f );

		return clamp( d, 0, 65535 );
	}
	
	inline static __device__ float UnpackDistance( unsigned distance, float truncationMargin )
	{
		return fmaf( distance / 32767.5f, truncationMargin, -truncationMargin );
	}
};

}} // namespace