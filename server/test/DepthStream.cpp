#include <boost/test/auto_unit_test.hpp>

#include <stdio.h>
#include <vector>

#include <DirectXMath.h>

#include <server/DepthFrame.h>
#include <server/DepthStream.h>

using namespace DirectX;



BOOST_AUTO_TEST_SUITE( DepthStream )

BOOST_AUTO_TEST_CASE( NextFrame )
{
	int const FRAME_RES = 640 * 480;
	char const * fileName = _tempnam( nullptr, nullptr );

	char const * header = "KPPL raw depth\n";
	int version = 1;
	int nFrames = 1;

	float viewMatrix[ 16 ] = { 0.0f };
	viewMatrix[ 3 + 1 * 4 ] = 0.271f;

	std::vector< short > depthData( FRAME_RES );
	depthData[ 71 + 103 * 640 ] = 57;

	FILE * file;
	fopen_s( & file, fileName, "wb" );

	fwrite( header, 1, 15, file );
	fwrite( & version, 4, 1, file );
	fwrite( & nFrames, 4, 1, file );
	fwrite( viewMatrix, sizeof( viewMatrix ), 1, file );
	fwrite( & depthData[ 0 ], 2, FRAME_RES, file );

	fclose( file );

	XMFLOAT4X4A view;
	kppl::DepthFrame depth;

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