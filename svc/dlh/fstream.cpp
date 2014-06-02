#include <fstream>
#include <string>

#include "fstream.h"



std::uint_least64_t dlh::fsize( std::string const & fileName )
{
	std::ifstream file;
	file.open( fileName.c_str(), std::ifstream::binary );
	file.seekg( 0, std::ifstream::end );
	return file.tellg();
}