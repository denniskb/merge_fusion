#include "DepthFrame.h"

#include <cassert>
#include <vector>



svc::DepthFrame::DepthFrame( int width, int height ) :
	m_width( 0 ),
	m_height( 0 )
{
	Resize( width, height );
}

void svc::DepthFrame::Resize( int newWidth, int newHeight )
{
	assert( newWidth >= 0 );
	assert( newHeight >= 0 );

	int newRes = newWidth * newHeight;
	if( Resolution() != newRes )
		m_data.resize( newRes );

	m_width = newWidth;
	m_height = newHeight;
}



int svc::DepthFrame::Width() const
{
	return m_width;
}

int svc::DepthFrame::Height() const
{
	return m_height;
}

int svc::DepthFrame::Resolution() const
{
	return Width() * Height();
}



float svc::DepthFrame::operator()( int x, int y ) const
{
	assert( x >= 0 );
	assert( y >= 0 );
	assert( x < Width() );
	assert( y < Height() );

	return m_data[ Index2Dto1D( x, y, Width() ) ];
}

float & svc::DepthFrame::operator()( int x, int y )
{
	assert( x >= 0 );
	assert( y >= 0 );
	assert( x < Width() );
	assert( y < Height() );

	return m_data[ Index2Dto1D( x, y, Width() ) ];
}



// static
int svc::DepthFrame::Index2Dto1D( int x, int y, int width )
{
	return x + y * width;
}