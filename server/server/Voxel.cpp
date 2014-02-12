#include "Voxel.h"

#include <algorithm>
#include <cassert>



kppl::Voxel::Voxel() :
	m_distance( (unsigned short) PackDistance( 0, 1.0f ) ), m_weight( 0 )
{
}



float kppl::Voxel::Distance( float truncationMargin ) const
{
	assert( truncationMargin > 0 );

	return UnpackDistance( m_distance, truncationMargin );
}

int kppl::Voxel::Weight() const
{
	return m_weight;
}



void kppl::Voxel::Update( float newDistance, float truncationMargin )
{
	assert( truncationMargin > 0 );

	if( Weight() < VOXEL_MAX_WEIGHT )
	{
		float d =
			( Weight() * Distance( truncationMargin ) + Clamp( newDistance, -truncationMargin, truncationMargin ) ) /
			( Weight() + 1 );

		m_distance = PackDistance( d, truncationMargin );
		m_weight++;
	}
}



bool kppl::Voxel::operator==( Voxel const & rhs ) const
{
	return
		m_distance == rhs.m_distance &&
		m_weight == rhs.m_weight;
}

bool kppl::Voxel::operator!=( Voxel const & rhs ) const
{
	return ! ( * this == rhs );
}



// static 
int kppl::Voxel::PackDistance( float distance, float truncationMargin )
{
	assert( truncationMargin > 0 );

	int d = (int) ( distance / truncationMargin * VOXEL_MAX_DIST_OVER_2 + VOXEL_MAX_DIST_OVER_2 + 0.5f ); // +0.5 => round to nearest int
	d = Clamp( d, 0, VOXEL_MAX_DIST );

	return d;
}

// static 
float kppl::Voxel::UnpackDistance( int distance, float truncationMargin )
{
	assert( truncationMargin > 0 );

	return ( distance - VOXEL_MAX_DIST_OVER_2 ) / VOXEL_MAX_DIST_OVER_2 * truncationMargin;
}



// static
int kppl::Voxel::Clamp( int x, int min, int max )
{
	assert( min <= max );

	x = std::max( min, x );
	x = std::min( max, x );

	return x;
}

// static
float kppl::Voxel::Clamp( float x, float min, float max )
{
	assert( min <= max );

	x = std::max( min, x );
	x = std::min( max, x );

	return x;
}