#include <cstdio>

#include <boost/test/auto_unit_test.hpp>

#include <dlh/fstream.h>


#pragma warning( disable : 4996 )

BOOST_AUTO_TEST_SUITE( fstream )

BOOST_AUTO_TEST_CASE( fsize )
{
	std::string const fileName = std::tmpnam( nullptr );
	
	{
		std::FILE * file = nullptr;
		if( ! fopen_s( & file, fileName.c_str(), "wb" ) && file )
		{
			fclose( file );	
			BOOST_CHECK( 0 == dlh::fsize( fileName ) );
		}
	}
	
	{
		std::FILE * file = nullptr;
		if( ! fopen_s( & file, fileName.c_str(), "wb" ) && file )
		{
			char data[1];
			fwrite( & data, 1, 1, file );
			fclose( file );
	
			BOOST_CHECK( 1 == dlh::fsize( fileName ) );
		}
	}

	std::remove( fileName.c_str() );
}

BOOST_AUTO_TEST_SUITE_END()