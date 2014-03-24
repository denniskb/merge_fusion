#include <boost/test/auto_unit_test.hpp>

#include <stdio.h>

#include <flink/util.h>
#include <flink/vector.h>



BOOST_AUTO_TEST_SUITE( util )

BOOST_AUTO_TEST_CASE( fsize )
{
	char const * fileName = _tempnam( nullptr, nullptr );
	
	FILE * file;
	fopen_s( & file, fileName, "wb" );
	fclose( file );
	
	BOOST_CHECK( 0 == flink::fsize( fileName ) );
	
	fopen_s( & file, fileName, "wb" );
	char data;
	fwrite( & data, 1, 1, file );
	fclose( file );
	
	BOOST_CHECK( 1 == flink::fsize( fileName ) );
	
	remove( fileName );

	std::free( (void*) fileName );
}

BOOST_AUTO_TEST_CASE( power2 )
{
	BOOST_REQUIRE( flink::powerOf2( 1 ) );
	BOOST_REQUIRE( flink::powerOf2( 2 ) );
	BOOST_REQUIRE( flink::powerOf2( 4 ) );
	BOOST_REQUIRE( flink::powerOf2( 2048 ) );
	BOOST_REQUIRE( flink::powerOf2( 65536 ) );
	BOOST_REQUIRE( flink::powerOf2( 2048 * 2048 ) );

	BOOST_REQUIRE( ! flink::powerOf2( 0 ) );
	BOOST_REQUIRE( ! flink::powerOf2( 3 ) );
	BOOST_REQUIRE( ! flink::powerOf2( 13 ) );
	BOOST_REQUIRE( ! flink::powerOf2( 26 ) );
	BOOST_REQUIRE( ! flink::powerOf2( 48 ) );
}

BOOST_AUTO_TEST_SUITE_END()