#include <cassert>
#include <fstream>
#include <string>

#include <kifi/util/math.h>
#include <kifi/util/fstream.h>
#include <kifi/util/vector2d.h>

#include <kifi/DepthStream.h>



namespace kifi {

DepthStream::DepthStream( std::string const & fileName ) :
	m_iFrame( 0 ),
	m_nFrames( 0 )
{
	assert( 4 == sizeof( int ) );
	assert( 4 == sizeof( DepthStream::TexelType ) );

	assert( util::fsize( fileName.c_str() ) >= 19 );

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



bool DepthStream::NextFrame
(
	util::vector2d< float > & outFrame,
	util::float4x4 & outView
)
{
	assert( 2 == sizeof( short ) );
	assert( 0 == outFrame.width() % 2 );

	if( m_iFrame >= m_nFrames )
		return false;

	float view[ 16 ];
	m_file.read( reinterpret_cast< char * >( view ), 64 );
	outView = util::transpose( util::float4x4( view ) ); // .depth uses row vectors, we use column vectors

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

			for( std::size_t i = 0, size = m_bufferedDepth.size() / 2; i < size; ++i )
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



int DepthStream::FrameCount() const
{
	return m_nFrames;
}

} // namespace