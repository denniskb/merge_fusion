#include "DepthStream.h"

#include <cassert>
#include <stdio.h>
#include <utility>
#include <vector>

#include "DepthFrame.h"
#include "flink.h"
#include "util.h"

using namespace flink;



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
	kppl::DepthFrame & outFrame,
	float4x4 & outView
)
{
	if( m_iFrame >= m_nFrames )
		return false;

	fread_s( outView.m, sizeof( outView.m ), 4, 16, m_file );

	outFrame.Resize( 640, 480 );
	fread_s( outFrame.data(), outFrame.Resolution() * 2, 2, outFrame.Resolution(), m_file );

	m_iFrame++;
	return true;
}