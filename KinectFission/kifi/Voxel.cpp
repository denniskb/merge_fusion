#include "Voxel.h"

#include <cassert>

#include <kifi/util/math.h>



namespace kifi {

Voxel::Voxel( unsigned data ) : m_data( data )
{
}

Voxel::operator unsigned() const
{
	return m_data;
}



float Voxel::Distance( float truncationMargin ) const
{
	assert( truncationMargin > 0.0f );

	return UnpackDistance( m_data >> 16, truncationMargin );
}

int Voxel::Weight() const
{
	return m_data & 0xff;
}



void Voxel::Update( float newDistance, float truncationMargin, int newWeight )
{
	assert( truncationMargin > 0.0f );

	unsigned w = Weight();
	unsigned w1 = w + newWeight;

	float d = 
	(
		w * Distance( truncationMargin ) + 
		newWeight * util::clamp( newDistance, -truncationMargin, truncationMargin ) 
	) / w1;

	m_data = PackDistance( d, truncationMargin ) << 16 | w1;
}



bool Voxel::operator==( Voxel rhs ) const
{
	return m_data == rhs.m_data;
}

bool Voxel::operator!=( Voxel rhs ) const
{
	return m_data != rhs.m_data;
}



// static 
unsigned Voxel::PackDistance( float distance, float truncationMargin )
{
	int d = (int) ( distance / truncationMargin * 32767.5f + 32768.0f );

	return util::clamp( d, 0, 65535 );
}
	
// static 
float Voxel::UnpackDistance( int distance, float truncationMargin )
{
	assert( truncationMargin > 0.0f );

	return distance / 32767.5f * truncationMargin - truncationMargin;
}

} // namespace