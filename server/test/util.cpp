#include <boost/test/auto_unit_test.hpp>

#include <stdio.h>

#include <server/util.h>



BOOST_AUTO_TEST_SUITE( util )

BOOST_AUTO_TEST_CASE( fsize )
{
	char const * fileName = _tempnam( nullptr, nullptr );

	FILE * file;
	fopen_s( & file, fileName, "wb" );
	fclose( file );

	BOOST_CHECK( 0 == kppl::fsize( fileName ) );

	fopen_s( & file, fileName, "wb" );
	char data;
	fwrite( & data, 1, 1, file );
	fclose( file );

	BOOST_CHECK( 1 == kppl::fsize( fileName ) );

	remove( fileName );
}

BOOST_AUTO_TEST_SUITE_END()