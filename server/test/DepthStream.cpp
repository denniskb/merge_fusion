#include <boost/test/auto_unit_test.hpp>

#include <stdio.h>
#include <vector>

#include <DirectXMath.h>

#include <server/HostDepthFrame.h>
#include <server/DepthStream.h>

using namespace DirectX;



static void WriteHeader
( 
	FILE * to,
	int version,
	int frameWidth, int frameHeight,
	int texelType,
	int nFrames
)
{
	if( version != 1 && version != 2 )
		BOOST_CHECK( false );

	static char const * header = "KPPL raw depth\n";
	fwrite( header, 1, 15, to );
	fwrite( & version, 4, 1, to );

	if( 2 == version )
	{
		fwrite( & frameWidth, 4, 1, to );
		fwrite( & frameHeight, 4, 1, to );
		fwrite( & texelType, 4, 1, to );
	}

	fwrite( & nFrames, 4, 1, to );
}



BOOST_AUTO_TEST_SUITE( DepthStream )

BOOST_AUTO_TEST_CASE( NextFrame_v1 )
{
	int const FRAME_RES = 640 * 480;
	char const * fileName = _tempnam( nullptr, nullptr );

	float viewMatrix[ 16 ] = { 0.0f };
	viewMatrix[ 3 + 1 * 4 ] = 0.271f;

	std::vector< short > depthData( FRAME_RES );
	depthData[ 71 + 103 * 640 ] = 57;

	FILE * file;
	fopen_s( & file, fileName, "wb" );

	WriteHeader( file, 1, -1, -1, -1, 1 );
	fwrite( viewMatrix, sizeof( viewMatrix ), 1, file );
	fwrite( & depthData[ 0 ], 2, FRAME_RES, file );

	fclose( file );

	XMFLOAT4X4A view;
	kppl::HostDepthFrame depth;

	kppl::DepthStream ds( fileName );
	BOOST_CHECK( ds.NextFrame( depth, view ) );
	BOOST_CHECK( 640 == depth.Width() );
	BOOST_CHECK( 480 == depth.Height() );
	BOOST_CHECK( 0 == depth( 0, 0 ) );
	BOOST_CHECK_CLOSE( 0.057f, depth( 71, 103 ), 0.1f );

	BOOST_CHECK( 0.0f == view._11 );
	BOOST_CHECK( 0.271f == view._24 );

	BOOST_CHECK( ! ds.NextFrame( depth, view ) );

	remove( fileName );
}

BOOST_AUTO_TEST_CASE( NextFrame_v2_short )
{
	int const FRAME_RES = 640 * 480;
	char const * fileName = _tempnam( nullptr, nullptr );

	float viewMatrix[ 16 ] = { 0.0f };
	viewMatrix[ 3 + 1 * 4 ] = 0.271f;

	std::vector< short > depthData( FRAME_RES );
	depthData[ 71 + 103 * 640 ] = 57;

	FILE * file;
	fopen_s( & file, fileName, "wb" );

	WriteHeader( file, 2, 640, 480, 0, 1 );
	fwrite( viewMatrix, sizeof( viewMatrix ), 1, file );
	fwrite( & depthData[ 0 ], 2, FRAME_RES, file );

	fclose( file );

	XMFLOAT4X4A view;
	kppl::HostDepthFrame depth;

	kppl::DepthStream ds( fileName );
	BOOST_CHECK( ds.NextFrame( depth, view ) );
	BOOST_CHECK( 640 == depth.Width() );
	BOOST_CHECK( 480 == depth.Height() );
	BOOST_CHECK( 0 == depth( 0, 0 ) );
	BOOST_CHECK_CLOSE( 0.057f, depth( 71, 103 ), 0.1f );

	BOOST_CHECK( 0.0f == view._11 );
	BOOST_CHECK( 0.271f == view._24 );

	BOOST_CHECK( ! ds.NextFrame( depth, view ) );

	remove( fileName );
}

BOOST_AUTO_TEST_CASE( NextFrame_v2_float )
{
	int const FRAME_RES = 640 * 480;
	char const * fileName = _tempnam( nullptr, nullptr );

	float viewMatrix[ 16 ] = { 0.0f };
	viewMatrix[ 3 + 1 * 4 ] = 0.271f;

	std::vector< float > depthData( FRAME_RES );
	depthData[ 71 + 103 * 640 ] = 0.057f;

	FILE * file;
	fopen_s( & file, fileName, "wb" );

	WriteHeader( file, 2, 640, 480, 1, 1 );
	fwrite( viewMatrix, sizeof( viewMatrix ), 1, file );
	fwrite( & depthData[ 0 ], 2, FRAME_RES, file );

	fclose( file );

	XMFLOAT4X4A view;
	kppl::HostDepthFrame depth;

	kppl::DepthStream ds( fileName );
	BOOST_CHECK( ds.NextFrame( depth, view ) );
	BOOST_CHECK( 640 == depth.Width() );
	BOOST_CHECK( 480 == depth.Height() );
	BOOST_CHECK( 0 == depth( 0, 0 ) );
	BOOST_CHECK_CLOSE( 0.057f, depth( 71, 103 ), 0.1f );

	BOOST_CHECK( 0.0f == view._11 );
	BOOST_CHECK( 0.271f == view._24 );

	BOOST_CHECK( ! ds.NextFrame( depth, view ) );

	remove( fileName );
}

BOOST_AUTO_TEST_SUITE_END()