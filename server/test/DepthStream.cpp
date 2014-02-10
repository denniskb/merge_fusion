#include <stdio.h>
#include <vector>

#include <boost/test/auto_unit_test.hpp>

#include <server/DepthStream.h>



BOOST_AUTO_TEST_SUITE( DepthStream )

BOOST_AUTO_TEST_CASE( NextFrame )
{
	int const FRAME_RES = 640 * 480;
	char const * fileName = _tempnam( nullptr, nullptr );

	FILE * file;
	fopen_s( & file, fileName, "wb" );

	char const * header = "KPPL raw depth\n";
	int version = 1;
	int nFrames = 1;
	char viewMatrix[ 64 ];
	std::vector< short > depthData( FRAME_RES );
	depthData[ 123 ] = 57;

	fwrite( header, 1, 15, file );
	fwrite( & version, 4, 1, file );
	fwrite( & nFrames, 4, 1, file );
	fwrite( viewMatrix, 1, sizeof( viewMatrix ), file );
	fwrite( & depthData[ 0 ], 2, FRAME_RES, file );

	fclose( file );

	kppl::DepthStream ds( fileName );
	depthData.clear();
	BOOST_CHECK( ds.NextFrame( depthData ) );
	BOOST_CHECK( FRAME_RES == depthData.size() );
	BOOST_CHECK( 0 == depthData[ 0 ] );
	BOOST_CHECK( 57 == depthData[ 123 ] );
	BOOST_CHECK( ! ds.NextFrame( depthData ) );

	remove( fileName );
}

BOOST_AUTO_TEST_SUITE_END()