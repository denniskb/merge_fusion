#include <boost/test/auto_unit_test.hpp>

#include <cstdio>

#include <flink/util.h>



BOOST_AUTO_TEST_SUITE( util )

BOOST_AUTO_TEST_CASE( fsize )
{
	char const * fileName = _tempnam( nullptr, nullptr );
	
	FILE * file = nullptr;
	if( ! fopen_s( & file, fileName, "wb" ) && file )
	{
		fclose( file );	
		BOOST_CHECK( 0 == flink::fsize( fileName ) );
	}
	
	if( ! fopen_s( & file, fileName, "wb" ) && file )
	{
		char data = 'X';
		fwrite( & data, 1, 1, file );
		fclose( file );
	
		BOOST_CHECK( 1 == flink::fsize( fileName ) );
	}

	remove( fileName );

	std::free( (void*) fileName );
}

BOOST_AUTO_TEST_SUITE_END()