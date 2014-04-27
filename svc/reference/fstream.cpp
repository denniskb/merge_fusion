#include "fstream.h"

#include <fstream>
#include <string>



long long svc::fsize( std::string const & fileName )
{
	std::ifstream file;
	file.open( fileName.c_str(), std::ifstream::binary );
	file.seekg( 0, std::ifstream::end );
	return file.tellg();
}