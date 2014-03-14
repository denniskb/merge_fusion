#pragma once

#include "cuda_directives.h"

#include <cassert>

#include <helper_math.h>



namespace kppl {

/*
A single cell as used inside an voxel volume,
capable of storing a weight and a signed distance.
The entire volume thus defines a signed distance field.
All parameters are in meters.
Measurements beyond truncationMargin in either direction are clamped.

This class is managed (.m), meaning it can be used by both host and device.
*/
class Voxel
{
public:
	kppl_host inline static int MAX_WEIGHT()
	{
		return 65535;
	}

	kppl_both inline Voxel() : m_data( 0 )
	{
	}

	kppl_both inline Voxel( unsigned data ) : m_data( data )
	{
	}

	kppl_both inline float Distance( float truncationMargin ) const
	{
		kppl_assert( truncationMargin > 0.0f );

		return UnpackDistance( m_data >> 16, truncationMargin );
	}
	
	kppl_both inline operator unsigned()
	{
		return m_data;
	}

	kppl_both inline int Weight() const
	{
		return m_data & 0xff;
	}

	kppl_both inline void Update( float newDistance, float truncationMargin, int newWeight = 1 )
	{
		kppl_assert( truncationMargin > 0.0f );

		unsigned w = Weight();
		unsigned w1 = w + newWeight;

		float d = ( w * Distance( truncationMargin ) + newWeight * clamp( newDistance, -truncationMargin, truncationMargin ) ) / w1;

		m_data = PackDistance( d, truncationMargin ) << 16 | w1;
	}

	/*
	Returns true iff this and rhs are bit-wise identical.
	*/
	kppl_host inline bool operator==( Voxel const rhs ) const
	{
		return m_data == rhs.m_data;
	}

	kppl_host inline bool operator!=( Voxel const rhs ) const
	{
		return m_data != rhs.m_data;
	}

	/*
	Returns true if both voxels have the same weight and their
	distances are not more then 'delta' apart
	*/
	kppl_host inline bool Close( Voxel const rhs, float truncationMargin, float delta ) const
	{
		return
			Weight() == rhs.Weight() &&
			std::abs( Distance( truncationMargin ) - rhs.Distance( truncationMargin ) ) <= delta;
	}

private:
	unsigned m_data;

	/*
	Maps distance from [-truncMargin, truncMargin] to [0, 63]
	*/
	kppl_both inline static unsigned PackDistance( float distance, float truncationMargin )
	{
		int d = (int) ( distance / truncationMargin * 32767.5f + 32768.0f );

		return clamp( d, 0, 65535 );
	}
	
	/*
	Maps distance from [0, 63] to [-truncMargin, truncMargin]
	*/
	kppl_both inline static float UnpackDistance( int distance, float truncationMargin )
	{
		kppl_assert( truncationMargin > 0.0f );

		return distance / 32767.5f * truncationMargin - truncationMargin;
	}
};

}