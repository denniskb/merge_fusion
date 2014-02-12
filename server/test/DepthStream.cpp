#include <stdio.h>
#include <vector>

#include <DirectXMath.h>

#include <boost/test/auto_unit_test.hpp>

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
	viewMatrix[ 7 ] = 0.271f;
	std::vector< short > depthData( FRAME_RES );
	depthData[ 123 ] = 57;

	FILE * file;
	fopen_s( & file, fileName, "wb" );

	fwrite( header, 1, 15, file );
	fwrite( & version, 4, 1, file );
	fwrite( & nFrames, 4, 1, file );
	fwrite( viewMatrix, sizeof( viewMatrix ), 1, file );
	fwrite( & depthData[ 0 ], 2, FRAME_RES, file );

	fclose( file );

	XMFLOAT4X4A view;
	depthData.clear();

	kppl::DepthStream ds( fileName );
	BOOST_CHECK( ds.NextFrame( depthData, view ) );
	BOOST_CHECK( FRAME_RES == depthData.size() );
	BOOST_CHECK( 0 == depthData[ 0 ] );
	BOOST_CHECK( 57 == depthData[ 123 ] );

	BOOST_CHECK( 0.0f == view._11 );
	BOOST_CHECK( 0.271f == view._24 );

	BOOST_CHECK( ! ds.NextFrame( depthData, view ) );

	remove( fileName );
}

BOOST_AUTO_TEST_SUITE_END()