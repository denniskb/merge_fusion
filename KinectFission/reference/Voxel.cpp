#include "Voxel.h"

#include <cassert>

#include <dlh/DirectXMathExt.h>



svc::Voxel::Voxel( unsigned data ) : m_data( data )
{
}

svc::Voxel::operator unsigned() const
{
	return m_data;
}



float svc::Voxel::Distance( float truncationMargin ) const
{
	assert( truncationMargin > 0.0f );

	return UnpackDistance( m_data >> 16, truncationMargin );
}

int svc::Voxel::Weight() const
{
	return m_data & 0xff;
}



void svc::Voxel::Update( float newDistance, float truncationMargin, int newWeight )
{
	assert( truncationMargin > 0.0f );

	unsigned w = Weight();
	unsigned w1 = w + newWeight;

	float d = 
	(
		w * Distance( truncationMargin ) + 
		newWeight * dlh::clamp( newDistance, -truncationMargin, truncationMargin ) 
	) / w1;

	m_data = PackDistance( d, truncationMargin ) << 16 | w1;
}



bool svc::Voxel::operator==( Voxel rhs ) const
{
	return m_data == rhs.m_data;
}

bool svc::Voxel::operator!=( Voxel rhs ) const
{
	return m_data != rhs.m_data;
}



// static 
unsigned svc::Voxel::PackDistance( float distance, float truncationMargin )
{
	int d = (int) ( distance / truncationMargin * 32767.5f + 32768.0f );

	return dlh::clamp( d, 0, 65535 );
}
	
// static 
float svc::Voxel::UnpackDistance( int distance, float truncationMargin )
{
	assert( truncationMargin > 0.0f );

	return distance / 32767.5f * truncationMargin - truncationMargin;
}