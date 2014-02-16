#include "DepthFrame.h"

#include <cassert>
#include <utility>
#include <vector>



kppl::DepthFrame::DepthFrame( int width, int height ) :
	m_width( width ),
	m_height( height )
{
	assert( m_width >= 0 );
	assert( m_height >= 0 );

	m_data.resize( width * height );
}

void kppl::DepthFrame::Resize( int newWidth, int newHeight )
{
	assert( newWidth >= 0 );
	assert( newHeight >= 0 );

	int newRes = newWidth * newHeight;
	if( newRes != Resolution() )
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



float kppl::DepthFrame::operator()( int x, int y ) const
{
	assert( x >= 0 && x < Width() );
	assert( y >= 0 && y < Height() );

	return m_data[ x + y * m_width ] * 0.001f;
}



short * kppl::DepthFrame::data()
{
	return & m_data[ 0 ];
}