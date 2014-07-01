#include <fstream>
#include <vector>

#include <boost/test/auto_unit_test.hpp>

#include <kifi/util/math.h>
#include <kifi/util/vector2d.h>

#include <kifi/DepthStream.h>

using namespace kifi;



static void WriteHeader
( 
	std::ofstream & to,
	std::int32_t version,
	std::int32_t frameWidth, std::int32_t frameHeight,
	std::int32_t texelType,
	std::int32_t nFrames
)
{
	if( version != 1 && version != 2 )
		BOOST_CHECK( false );

	to.write( "svc raw depth\n", 15 );
	to.write( reinterpret_cast< char * >( & version ), 4 );

	if( 2 == version )
	{
		to.write( reinterpret_cast< char * >( & frameWidth  ), 4 );
		to.write( reinterpret_cast< char * >( & frameHeight ), 4 );
		to.write( reinterpret_cast< char * >( & texelType   ), 4 );
	}

	to.write( reinterpret_cast< char * >( & nFrames ), 4 );
}



BOOST_AUTO_TEST_SUITE( DepthStreamTest )

BOOST_AUTO_TEST_CASE( NextFrame_v1 )
{
	int const FRAME_RES = 640 * 480;
	char * fileName = _tempnam( nullptr, nullptr );

	float viewMatrix[ 16 ] = { 0.0f };
	viewMatrix[ 2 + 1 * 4 ] = 0.271f;

	std::vector< std::int16_t > depthData( FRAME_RES );
	depthData[ 71 + 103 * 640 ] = 57;

	std::ofstream file( fileName, std::ofstream::binary );
	WriteHeader( file, 1, -1, -1, -1, 1 );
	file.write( reinterpret_cast< char * >( & viewMatrix ), sizeof( viewMatrix ) );
	file.write( reinterpret_cast< char * >( depthData.data() ), 2 * depthData.size() );
	file.close();

	util::matrix4x3 view;
	util::vector2d< float > depth;

	DepthStream ds( fileName );
	BOOST_CHECK( ds.NextFrame( depth, view ) );
	BOOST_CHECK( 640 == depth.width() );
	BOOST_CHECK( 480 == depth.height() );
	BOOST_CHECK( 0 == depth( 0, 0 ) );
	BOOST_CHECK_CLOSE( 0.057f, depth( 71, 103 ), 0.1f );

	BOOST_CHECK( 0.0f == view.m00 );
	BOOST_CHECK( 0.271f == view.m12 );

	BOOST_CHECK( ! ds.NextFrame( depth, view ) );

	remove( fileName );
	std::free( fileName );
}

BOOST_AUTO_TEST_CASE( NextFrame_v2_short )
{
	int const FRAME_RES = 640 * 480;
	char * fileName = _tempnam( nullptr, nullptr );

	float viewMatrix[ 16 ] = { 0.0f };
	viewMatrix[ 2 + 1 * 4 ] = 0.271f;

	std::vector< std::int16_t > depthData( FRAME_RES );
	depthData[ 71 + 103 * 640 ] = 57;

	std::ofstream file( fileName, std::ofstream::binary );
	WriteHeader( file, 2, 640, 480, 0, 1 );
	file.write( reinterpret_cast< char * >( & viewMatrix ), sizeof( viewMatrix ) );
	file.write( reinterpret_cast< char * >( depthData.data() ), 2 * depthData.size() );
	file.close();

	util::matrix4x3 view;
	util::vector2d< float > depth;

	DepthStream ds( fileName );
	BOOST_CHECK( ds.NextFrame( depth, view ) );
	BOOST_CHECK( 640 == depth.width() );
	BOOST_CHECK( 480 == depth.height() );
	BOOST_CHECK( 0 == depth( 0, 0 ) );
	BOOST_CHECK_CLOSE( 0.057f, depth( 71, 103 ), 0.1f );

	BOOST_CHECK( 0.0f == view.m00 );
	BOOST_CHECK( 0.271f == view.m12 );

	BOOST_CHECK( ! ds.NextFrame( depth, view ) );

	remove( fileName );
	std::free( fileName );
}

BOOST_AUTO_TEST_CASE( NextFrame_v2_float )
{
	int const FRAME_RES = 640 * 480;
	char * fileName = _tempnam( nullptr, nullptr );

	float viewMatrix[ 16 ] = { 0.0f };
	viewMatrix[ 2 + 1 * 4 ] = 0.271f;

	std::vector< float > depthData( FRAME_RES );
	depthData[ 71 + 103 * 640 ] = 0.057f;

	std::ofstream file( fileName, std::ofstream::binary );
	WriteHeader( file, 2, 640, 480, 1, 1 );
	file.write( reinterpret_cast< char * >( & viewMatrix ), sizeof( viewMatrix ) );
	file.write( reinterpret_cast< char * >( depthData.data() ), 4 * depthData.size() );
	file.close();

	util::matrix4x3 view;
	util::vector2d< float > depth;

	DepthStream ds( fileName );
	BOOST_CHECK( ds.NextFrame( depth, view ) );
	BOOST_CHECK( 640 == depth.width() );
	BOOST_CHECK( 480 == depth.height() );
	BOOST_CHECK( 0 == depth( 0, 0 ) );
	BOOST_CHECK_CLOSE( 0.057f, depth( 71, 103 ), 0.1f );

	BOOST_CHECK( 0.0f == view.m00 );
	BOOST_CHECK( 0.271f == view.m12 );

	BOOST_CHECK( ! ds.NextFrame( depth, view ) );

	remove( fileName );
	std::free( fileName );
}

BOOST_AUTO_TEST_SUITE_END()