#include <fstream>
#include <string>

#include "fstream.h"



namespace kifi {
namespace util {

std::uint_least64_t fsize( std::string const & fileName )
{
	std::ifstream file;
	file.open( fileName.c_str(), std::ifstream::binary );
	file.seekg( 0, std::ifstream::end );
	return file.tellg();
}

}} // namespace