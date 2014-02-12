#include "DepthStream.h"

#include <cassert>
#include <stdio.h>

#include "util.h"

using namespace DirectX;



kppl::DepthStream::DepthStream( char const * fileName ) :
	m_iFrame( 0 )
{
	assert( kppl::fsize( fileName ) >= 19 );

	fopen_s( & m_file, fileName, "rb" );
	_fseeki64( m_file, 15, SEEK_SET ); // skip magic

	int version;
	fread_s( & version, sizeof( version ), 4, 1, m_file );
	assert( 1 == version );

	fread_s( & m_nFrames, sizeof( m_nFrames ), 4, 1, m_file );
}

kppl::DepthStream::~DepthStream()
{
	fclose( m_file );
}



bool kppl::DepthStream::NextFrame
(
	std::vector< short > & outFrame,
	XMFLOAT4X4A & outView
)
{
	if( m_iFrame >= m_nFrames )
		return false;

	fread_s( outView.m, sizeof( outView.m ), 4, 16, m_file );

	outFrame.resize( 640 * 480 );
	fread_s( & outFrame[ 0 ], outFrame.size() * 2, 2, 640 * 480, m_file );
	m_iFrame++;

	return true;
}