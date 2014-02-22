#include <boost/test/auto_unit_test.hpp>

#pragma warning( push )
#pragma warning( disable: 4100 4267 4515 4996 )
#include <server/DeviceVolume.h>
#pragma warning( pop )

#include <server/HostVolume.h>
#include <server/Voxel.h>



BOOST_AUTO_TEST_SUITE( DeviceVolume )

BOOST_AUTO_TEST_CASE( copy )
{
	kppl::HostVolume hvol1( 8, 1.0f, 0.01f );
	hvol1( 2, 7, 1 ).Update( 0.005f, 0.01f );

	kppl::DeviceVolume dvol( hvol1 );
	
	kppl::HostVolume hvol2( 8, 1.0f, 0.01f );
	dvol >> hvol2;

	BOOST_REQUIRE( 1 == hvol2( 2, 7, 1 ).Weight() );
	BOOST_REQUIRE_CLOSE( 0.005f, hvol2( 2, 7, 1 ).Distance( 0.01f ), 0.1f );
}

BOOST_AUTO_TEST_CASE( assign )
{
	kppl::HostVolume hvol1( 8, 1.0f, 0.01f );
	kppl::DeviceVolume dvol( hvol1 );

	kppl::HostVolume hvol2( 8, 1.0f, 0.01f );
	hvol2( 2, 7, 1 ).Update( 0.005f, 0.01f );

	dvol << hvol2;
	dvol >> hvol1;

	BOOST_REQUIRE( 1 == hvol1( 2, 7, 1 ).Weight() );
	BOOST_REQUIRE_CLOSE( 0.005f, hvol1( 2, 7, 1 ).Distance( 0.01f ), 0.1f );
}

BOOST_AUTO_TEST_SUITE_END()