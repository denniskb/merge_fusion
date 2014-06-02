#include <cassert>
#include <fstream>
#include <string>

#include <dlh/DirectXMathExt.h>
#include <dlh/fstream.h>
#include <dlh/vector2d.h>

#include "DepthStream.h"



svc::DepthStream::DepthStream( std::string const & fileName ) :
	m_iFrame( 0 ),
	m_nFrames( 0 )
{
	assert( 4 == sizeof( int ) );
	assert( 4 == sizeof( DepthStream::TexelType ) );

	assert( dlh::fsize( fileName.c_str() ) >= 19 );

	m_file.open( fileName.c_str(), std::ifstream::binary );
	m_file.seekg( 15 ); // skip magic

	int version;
	m_file.read( reinterpret_cast< char * >( & version ), 4 );

	switch( version )
	{
	case 1:
		m_frameWidth = 640;
		m_frameHeight = 480;
		m_texelType = SHORT;
		break;

	case 2:
		m_file.read( reinterpret_cast< char * >( & m_frameWidth ), 4 );
		m_file.read( reinterpret_cast< char * >( & m_frameHeight ), 4 );
		m_file.read( reinterpret_cast< char * >( & m_texelType ), 4 );
		break;

	default:
		assert( false );
		return;
	}

	m_file.read( reinterpret_cast< char * >( & m_nFrames ), 4 );

	m_bufferedDepth.resize( m_frameWidth, m_frameHeight );
}



bool svc::DepthStream::NextFrame
(
	dlh::vector2d< float > & outFrame,
	dlh::float4x4 & outView
)
{
	assert( 2 == sizeof( short ) );
	assert( 0 == outFrame.width() % 2 );

	if( m_iFrame >= m_nFrames )
		return false;

	m_file.read( reinterpret_cast< char * >( & outView.m ), 64 );

	outFrame.resize( m_frameWidth, m_frameHeight );

	switch( m_texelType )
	{
	case FLOAT:
		m_file.read( reinterpret_cast< char * >( outFrame.data() ), outFrame.size() * 4 );
		break;

	case SHORT:
		{
			m_file.read( reinterpret_cast< char * >( m_bufferedDepth.data() ), m_bufferedDepth.size() * 2 );
			unsigned * in = reinterpret_cast< unsigned * >( m_bufferedDepth.data() );

			for( size_t i = 0, size = m_bufferedDepth.size() / 2; i < size; ++i )
			{
				// little endian
				outFrame[ 2 * i ] = ( in[ i ] & 0xffff ) * 0.001f;
				outFrame[ 2 * i + 1 ] = ( in[ i ] >> 16 ) * 0.001f;
			}
		}
		break;

	default:
		assert( false );
		return false;
	}

	++m_iFrame;
	return true;
}