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
		return 15;
	}

	kppl_host inline static float PACK_SCALE( float truncationMargin )
	{
		assert( truncationMargin > 0.0f );

		return 2047.5f / truncationMargin;
	}

	kppl_host inline static float UNPACK_SCALE( float truncationMargin )
	{
		assert( truncationMargin > 0.0f );

		return truncationMargin / 2047.5f;
	}

	kppl_host inline Voxel() : m_data( 0 )
	{
	}

	kppl_host inline float Distance( float truncationMargin ) const
	{
		assert( truncationMargin > 0.0f );

		return Distance( truncationMargin, UNPACK_SCALE( truncationMargin ) );
	}
	
	/*
	Faster version of Distance( float ), use in performance-critical code.
	Use values obtained from Voxel::(UN)PACK_SCALE for (un)packScale
	*/
	kppl_both inline float Distance( float truncationMargin, float unpackScale ) const
	{
		kppl_assert( truncationMargin > 0.0f );
		kppl_assert( unpackScale > 0.0f );

		return UnpackDistance( m_data >> 4, truncationMargin, unpackScale );
	}
	
	kppl_both inline int Weight() const
	{
		return m_data & 0xf;
	}

	kppl_host inline void Update( float newDistance, float truncationMargin )
	{
		assert( truncationMargin > 0.0f );

		Update( newDistance, truncationMargin, PACK_SCALE( truncationMargin ), UNPACK_SCALE( truncationMargin ) );
	}

	/*
	Faster version of Update( float, float ), use in performance-critical code.
	Use values obtained from Voxel::(UN)PACK_SCALE for (un)packScale
	*/
	kppl_both inline void Update
	( 
		float newDistance, 
		float truncationMargin,
		float packScale,
		float unpackScale
	)
	{
		kppl_assert( truncationMargin > 0.0f );
		kppl_assert( packScale > 0.0f );
		kppl_assert( unpackScale > 0.0f );

		unsigned w = Weight();
		unsigned w1 = w + 1;

		float d = ( w * Distance( truncationMargin, unpackScale ) + clamp( newDistance, -truncationMargin, truncationMargin ) ) / w1;

		m_data = (unsigned short) ( PackDistance( d, packScale ) << 4 | w1 );
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
	unsigned short m_data;

	/*
	Maps distance from [-truncMargin, truncMargin] to [0, 63]
	*/
	kppl_both inline static unsigned PackDistance( float distance, float packScale )
	{
		kppl_assert( packScale > 0.0f );

#ifdef __CUDA_ARCH__
		int d = (int) fmaf( distance, packScale, 2048.0f );
#else
		int d = (int) ( distance * packScale + 2048.0f );
#endif

		return clamp( d, 0, 4095 );
	}
	
	/*
	Maps distance from [0, 63] to [-truncMargin, truncMargin]
	*/
	kppl_both inline static float UnpackDistance( int distance, float truncationMargin, float unpackScale )
	{
		kppl_assert( truncationMargin > 0.0f );
		kppl_assert( unpackScale > 0.0f );

#ifdef __CUDA_ARCH__
		return fmaf( distance, unpackScale, -truncationMargin );
#else
		return distance * unpackScale - truncationMargin;
#endif
	}
};

}