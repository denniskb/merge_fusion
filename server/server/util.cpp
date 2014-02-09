#include "util.h"

#include <stdio.h>



long long kppl::fsize( char const * fileName )
{
	FILE * test;
	fopen_s( & test, fileName, "rb" );

	_fseeki64( test, 0, SEEK_END );
	long long result = _ftelli64( test );

	fclose( test );

	return result;
} 