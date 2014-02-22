#include "HostDepthFrame.h"

#include <cassert>
#include <utility>
#include <vector>



kppl::HostDepthFrame::HostDepthFrame( int width, int height ) :
	m_width( 0 ),
	m_height( 0 )
{
	Resize( width, height );
}

void kppl::HostDepthFrame::Resize( int newWidth, int newHeight )
{
	assert( newWidth >= 0 );
	assert( newHeight >= 0 );

	int newRes = newWidth * newHeight;
	if( Resolution() != newRes )
		m_data.resize( newRes );

	m_width = newWidth;
	m_height = newHeight;
}



int kppl::HostDepthFrame::Width() const
{
	return m_width;
}

int kppl::HostDepthFrame::Height() const
{
	return m_height;
}

int kppl::HostDepthFrame::Resolution() const
{
	return Width() * Height();
}



float & kppl::HostDepthFrame::operator()( int x, int y )
{
	return m_data[ Index2Dto1D( x, y, Width(), Height() ) ];
}

float const & kppl::HostDepthFrame::operator()( int x, int y ) const
{
	return m_data[ Index2Dto1D( x, y, Width(), Height() ) ];
}



#pragma warning( push )
#pragma warning( disable: 4100 ) // height = unreferenced param in release build
// static
int kppl::HostDepthFrame::Index2Dto1D( int x, int y, int width, int height )
{
	assert( x >= 0 );
	assert( y >= 0 );
	assert( x < width );
	assert( y < height );

	return x + y * width;
}
#pragma warning( pop )