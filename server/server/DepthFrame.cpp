#include "DepthFrame.h"

#include <cassert>
#include <utility>
#include <vector>



kppl::DepthFrame::DepthFrame( int width, int height ) :
	m_width( 0 ),
	m_height( 0 )
{
	Resize( width, height );
}

void kppl::DepthFrame::Resize( int newWidth, int newHeight )
{
	assert( newWidth >= 0 );
	assert( newHeight >= 0 );

	int newRes = newWidth * newHeight;
	if( Resolution() != newRes )
		m_data.resize( newRes );

	m_width = newWidth;
	m_height = newHeight;
}



int kppl::DepthFrame::Width() const
{
	return m_width;
}

int kppl::DepthFrame::Height() const
{
	return m_height;
}

int kppl::DepthFrame::Resolution() const
{
	return Width() * Height();
}



float & kppl::DepthFrame::operator()( int x, int y )
{
	return m_data[ Index2Dto1D( x, y ) ];
}

float const & kppl::DepthFrame::operator()( int x, int y ) const
{
	return m_data[ Index2Dto1D( x, y ) ];
}



int kppl::DepthFrame::Index2Dto1D( int x, int y ) const
{
	assert( x >= 0 );
	assert( y >= 0 );
	assert( x < Width() );
	assert( y < Height() );

	return x + y * Width();
}