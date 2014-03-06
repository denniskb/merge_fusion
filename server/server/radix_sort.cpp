#include "radix_sort.h"

#include <cstring>
#include <vector>



void kppl::radix_sort
(
	std::vector< unsigned > & A,
	std::vector< unsigned > & B
)
{
	unsigned cnt[ 256 ];
	unsigned const mask = 0xff;

	B.resize( A.size() );

	for( int shift = 0; shift < 32; shift += 8 )
	{
		std::memset( cnt, 0, 1024 );

		for( int i = 0; i < A.size(); i++ )
			cnt[ ( A[ i ] >> shift ) & mask ]++;

		for( int i = 1; i < 256; i++ )
			cnt[ i ] += cnt[ i - 1 ];

		for( int i = (int) ( A.size() - 1 ); i >= 0; i-- )
			B[ --cnt[ ( A[ i ] >> shift ) & mask ] ] = A[ i ];

		std::swap( A, B );
	}
}