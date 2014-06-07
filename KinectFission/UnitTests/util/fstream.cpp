#include <cstdio>

#include <boost/test/auto_unit_test.hpp>

#include <kifi/util/fstream.h>

using namespace kifi;

// function tmpnam may be unsafe
#pragma warning( disable : 4996 )



BOOST_AUTO_TEST_SUITE( util_test )
BOOST_AUTO_TEST_SUITE( fstream_test )

BOOST_AUTO_TEST_CASE( fsize )
{
	std::string const fileName = std::tmpnam( nullptr );
	
	{
		std::FILE * file = nullptr;
		if( ! fopen_s( & file, fileName.c_str(), "wb" ) && file )
		{
			fclose( file );	
			BOOST_CHECK( 0 == util::fsize( fileName ) );
		}
	}
	
	{
		std::FILE * file = nullptr;
		if( ! fopen_s( & file, fileName.c_str(), "wb" ) && file )
		{
			char data[1];
			fwrite( & data, 1, 1, file );
			fclose( file );
	
			BOOST_CHECK( 1 == util::fsize( fileName ) );
		}
	}

	std::remove( fileName.c_str() );
}

BOOST_AUTO_TEST_SUITE_END()
BOOST_AUTO_TEST_SUITE_END()