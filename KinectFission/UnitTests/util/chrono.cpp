#include <chrono>
#include <thread>

#include <boost/test/auto_unit_test.hpp>

#include <kifi/util/chrono.h>

using namespace kifi;



BOOST_AUTO_TEST_SUITE( chrono )

BOOST_AUTO_TEST_CASE( hires_clock )
{
	auto t1 = util::chrono::high_resolution_clock::now();
	auto t2 = util::chrono::high_resolution_clock::now();

	BOOST_REQUIRE( ( t2 - t1 ).count() >= 0 );
}

BOOST_AUTO_TEST_CASE( stop_watch )
{
	// Quick visual test
	std::printf( "Entering test case \"stop_watch\"\n" );

	util::chrono::stop_watch sw;

	std::this_thread::sleep_for( std::chrono::milliseconds( 60 ) );
	sw.take_time( "first run (60ms)" );

	std::this_thread::sleep_for( std::chrono::milliseconds( 30 ) );
	sw.take_time( "second run (30ms)" );

	sw.print_times();

	std::printf( "Leaving test case \"stop_watch\"\n" );
}

BOOST_AUTO_TEST_SUITE_END()