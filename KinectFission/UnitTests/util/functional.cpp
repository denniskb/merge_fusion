#include <cstdlib>

#include <boost/test/auto_unit_test.hpp>

#include <kifi/util/functional.h>

using namespace kifi;



BOOST_AUTO_TEST_SUITE( util_test )
BOOST_AUTO_TEST_SUITE( functional_test )

BOOST_AUTO_TEST_CASE( id )
{
	util::id< int > op;

	for( int i = 0; i < 100; i++ )
	{
		int const x = std::rand();
		BOOST_REQUIRE( op( x ) == x );
	}
}

BOOST_AUTO_TEST_SUITE_END()
BOOST_AUTO_TEST_SUITE_END()