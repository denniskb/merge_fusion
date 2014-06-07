#include <atomic>
#include <chrono>
#include <thread>

#include <boost/test/auto_unit_test.hpp>

#include <kifi/util/barrier.h>

using namespace std::chrono;
using namespace kifi;



static void sleep_for( int ms )
{
	std::this_thread::sleep_for( std::chrono::milliseconds( ms ) );
}



BOOST_AUTO_TEST_SUITE( barrier )

BOOST_AUTO_TEST_CASE( basic )
{
	util::barrier bar( 2 );
	std::atomic< int > count( 0 );

	std::thread worker( [&]()
	{
		BOOST_CHECK( 0 == count );
		bar.wait( 1 );
		count++;
	});

	sleep_for( 50 );
	BOOST_CHECK( 0 == count );
	bar.wait( 0 );
	count++;

	worker.join();

	BOOST_CHECK( 2 == count );
}

BOOST_AUTO_TEST_CASE( multi_pass )
{
	util::barrier bar( 2 );
	std::atomic< int > count( 0 );

	std::thread worker( [&]()
	{
		BOOST_CHECK( 0 == count );
		bar.wait( 1 );
		count++;
		bar.wait( 1 );
		count++;
	});

	bar.wait( 0 );
	sleep_for( 50 );
	BOOST_CHECK( count <= 1 );

	bar.wait( 0 );
	worker.join();

	BOOST_CHECK( 2 == count );
}

BOOST_AUTO_TEST_SUITE_END()