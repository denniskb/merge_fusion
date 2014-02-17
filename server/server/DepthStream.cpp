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
	assert( 1 == version || 2 == version );

	if( 2 == version )
	{
		fread_s( & m_frameWidth, sizeof( m_frameWidth ), 4, 1, m_file );
		fread_s( & m_frameHeight, sizeof( m_frameHeight ), 4, 1, m_file );

		int texelType = -1;
		fread_s( & texelType, sizeof( texelType ), 4, 1, m_file );
		m_texelType = static_cast< TexelType >( texelType );
	}
	else
	{
		m_frameWidth = 640;
		m_frameHeight = 480;
		m_texelType = SHORT;
	}

	fread_s( & m_nFrames, sizeof( m_nFrames ), 4, 1, m_file );

	m_bufferDepth.resize( m_frameWidth * m_frameHeight );
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

	outFrame.Resize( m_frameWidth, m_frameHeight );
	switch( m_texelType )
	{
	case FLOAT:
		fread_s( & outFrame( 0, 0 ), outFrame.Resolution() * 4, 4, outFrame.Resolution(), m_file );
		break;

	case SHORT:
		fread_s( & m_bufferDepth[ 0 ], m_bufferDepth.size() * 2, 2, m_bufferDepth.size(), m_file );
		for( int y = 0; y < outFrame.Height(); y++ )
			for( int x = 0; x < outFrame.Width(); x++ )
				outFrame( x, y ) = m_bufferDepth[ x + y * outFrame.Width() ] * 0.001f;

		break;

	default:
		assert( false );
	}

	m_iFrame++;
	return true;
}