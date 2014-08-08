#include <iterator>

#include <boost/test/auto_unit_test.hpp>

#include <kifi/util/functional.h>
#include <kifi/util/iterator.h>

using namespace kifi;



BOOST_AUTO_TEST_SUITE( util_test )
BOOST_AUTO_TEST_SUITE( iterator_test )

BOOST_AUTO_TEST_CASE( const_iterator )
{
	util::const_iterator< int > x( 5 );
	util::const_iterator< int > y( 6 );

	BOOST_REQUIRE( 5 == * x );
	BOOST_REQUIRE( 6 == * y );

	++x;
	++y;

	BOOST_REQUIRE( 5 == * x );
	BOOST_REQUIRE( 6 == * y );
}

BOOST_AUTO_TEST_CASE( transform_iterator )
{
	int const a[] = { 1, 3, 3, 7, 0 };
	int const * pa = a;
	
	auto first = util::make_transform_iterator( a, util::offset< int >( 5 ) );
	auto last  = util::make_transform_iterator( a + 5, util::offset< int >( 5 ) );

	BOOST_REQUIRE( 5 == std::distance( first, last ) );

	for( auto it = first; it != last; ++it, ++pa )
		BOOST_REQUIRE( * it == * pa + 5 );
}

BOOST_AUTO_TEST_SUITE_END()
BOOST_AUTO_TEST_SUITE_END()