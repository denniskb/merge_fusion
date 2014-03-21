#include <boost/test/auto_unit_test.hpp>

#include <stdio.h>

#include <reference/util.h>
#include <reference/vector.h>



BOOST_AUTO_TEST_SUITE( util )

BOOST_AUTO_TEST_CASE( fsize )
{
	char const * fileName = _tempnam( nullptr, nullptr );

	FILE * file;
	fopen_s( & file, fileName, "wb" );
	fclose( file );

	BOOST_CHECK( 0 == svc::fsize( fileName ) );

	fopen_s( & file, fileName, "wb" );
	char data;
	fwrite( & data, 1, 1, file );
	fclose( file );

	BOOST_CHECK( 1 == svc::fsize( fileName ) );

	remove( fileName );
}

BOOST_AUTO_TEST_CASE( remove_dups )
{
	svc::vector< unsigned > test;
	test.push_back( 1 );
	test.push_back( 1 );
	test.push_back( 2 );

	svc::remove_dups( test );

	BOOST_REQUIRE( test.size() == 2 );
	BOOST_REQUIRE( test[ 0 ] == 1 );
	BOOST_REQUIRE( test[ 1 ] == 2 );
}

BOOST_AUTO_TEST_CASE( remove_value )
{
	svc::vector< unsigned > test;
	test.push_back( 1 );
	test.push_back( 1 );
	test.push_back( 2 );

	svc::remove_value( test, 1 );

	BOOST_REQUIRE( test.size() == 1 );
	BOOST_REQUIRE( test[ 0 ] == 2 );
}

BOOST_AUTO_TEST_CASE( power2 )
{
	BOOST_REQUIRE( svc::powerOf2( 1 ) );
	BOOST_REQUIRE( svc::powerOf2( 2 ) );
	BOOST_REQUIRE( svc::powerOf2( 4 ) );
	BOOST_REQUIRE( svc::powerOf2( 2048 ) );
	BOOST_REQUIRE( svc::powerOf2( 65536 ) );
	BOOST_REQUIRE( svc::powerOf2( 2048 * 2048 ) );

	BOOST_REQUIRE( ! svc::powerOf2( 0 ) );
	BOOST_REQUIRE( ! svc::powerOf2( 3 ) );
	BOOST_REQUIRE( ! svc::powerOf2( 13 ) );
	BOOST_REQUIRE( ! svc::powerOf2( 26 ) );
	BOOST_REQUIRE( ! svc::powerOf2( 48 ) );
}

BOOST_AUTO_TEST_SUITE_END()