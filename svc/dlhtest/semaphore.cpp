#include <atomic>
#include <chrono>
#include <thread>

#include <boost/test/auto_unit_test.hpp>

#include <dlh/semaphore.h>

using namespace std::chrono;



static void sleep_for( int ms )
{
	std::this_thread::sleep_for( std::chrono::milliseconds( ms ) );
}



BOOST_AUTO_TEST_SUITE( semaphore )

BOOST_AUTO_TEST_CASE( basic_wait_signal )
{
	dlh::semaphore sem( 0 );
	int critical = 0;

	std::thread worker( [&]()
	{
		sem.wait();
		BOOST_CHECK( 1 == critical );
	});

	sleep_for( 50 );
	BOOST_CHECK( 0 == critical );
	critical = 1;
	sem.signal();

	worker.join();
}

BOOST_AUTO_TEST_CASE( multi_signal )
{
	dlh::semaphore sem( 0 );
	std::atomic< int > critical( 0 );

	std::thread worker1( [&]()
	{
		sem.wait();
		critical++;
	});

	std::thread worker2( [&]()
	{
		sem.wait();
		critical++;
	});

	sleep_for( 50 );
	BOOST_CHECK( 0 == critical );
	sem.signal( 2 );
	worker1.join();
	worker2.join();
	BOOST_CHECK( 2 == critical );
}

BOOST_AUTO_TEST_CASE( preload )
{
	dlh::semaphore sem( 1 );

	sem.wait();

	BOOST_CHECK( true );
}

BOOST_AUTO_TEST_CASE( negative_preload )
{
	dlh::semaphore sem( -1 );
	std::atomic< int > critical( 0 );

	std::thread worker1( [&]()
	{
		critical++;
		sem.signal();
	});

	std::thread worker2( [&]()
	{
		sleep_for( 50 );
		critical++;
		sem.signal();
	});

	sem.wait();
	BOOST_CHECK( 2 == critical );

	worker1.join();
	worker2.join();
}

BOOST_AUTO_TEST_SUITE_END()