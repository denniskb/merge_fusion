#include <boost/test/auto_unit_test.hpp>

#pragma warning( push )
#pragma warning( disable: 4100 4267 4515 4996 )
#include <server/DeviceDepthFrame.h>
#pragma warning( pop )

#include <server/HostDepthFrame.h>



BOOST_AUTO_TEST_SUITE( DeviceDepthFrame )

BOOST_AUTO_TEST_CASE( copy )
{
	kppl::HostDepthFrame hdf( 256, 256 );
	hdf( 57, 209 ) = 2.13f;
	
	kppl::DeviceDepthFrame ddf( hdf );
	hdf( 57, 209 ) = 0.0f;
	ddf.CopyTo( hdf );

	BOOST_REQUIRE_CLOSE( 2.13f, hdf( 57, 209 ), 0.1f );
}

BOOST_AUTO_TEST_CASE( assign )
{
	kppl::HostDepthFrame hdf1;
	kppl::DeviceDepthFrame ddf( hdf1 );

	kppl::HostDepthFrame hdf2( 128, 128 );
	ddf = hdf2;

	ddf.CopyTo( hdf1 );

	BOOST_REQUIRE( 128 == hdf1.Width() );
	BOOST_REQUIRE( 128 == hdf1.Height() );
}

BOOST_AUTO_TEST_SUITE_END()