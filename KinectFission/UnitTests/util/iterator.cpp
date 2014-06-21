#include <iterator>

#include <boost/test/auto_unit_test.hpp>

#include <kifi/util/iterator.h>

using namespace kifi;



namespace {

struct delta
{
	int n;

	delta( int n ) : n( n ) {}
	int operator()( int x ) const { return x + n; }
};

}



BOOST_AUTO_TEST_SUITE( util_test )
BOOST_AUTO_TEST_SUITE( iterator_test )

BOOST_AUTO_TEST_CASE( partial_sum_exclusive )
{
	int const a[] = { 1, 3, 3, 7, 0 };
	int const * pa = a;
	
	auto first = util::make_transform_iterator( a, delta( 5 ) );
	auto last  = util::make_transform_iterator( a + 5, delta( 5 ) );

	BOOST_REQUIRE( 5 == std::distance( first, last ) );

	for( auto it = first; it != last; ++it, ++pa )
		BOOST_REQUIRE( * it == * pa + 5 );
}

BOOST_AUTO_TEST_SUITE_END()
BOOST_AUTO_TEST_SUITE_END()