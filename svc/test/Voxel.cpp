#include <boost/test/auto_unit_test.hpp>

#include <reference/Voxel.m>



BOOST_AUTO_TEST_SUITE( Voxel )

BOOST_AUTO_TEST_CASE( Update )
{
	float const tm = 0.271f;

	svc::Voxel v;
	BOOST_REQUIRE( 0 == v.Weight() );

	v.Update( 0.1f, tm );
	BOOST_REQUIRE_CLOSE( v.Distance( tm ), 0.1f, 0.1f );
	BOOST_REQUIRE( 1 == v.Weight() );

	svc::Voxel v2;
	v2.Update( 0.5f, tm );
	BOOST_REQUIRE_CLOSE( v2.Distance( tm ), tm, 0.1f );
}

BOOST_AUTO_TEST_CASE( UpdateWeight )
{
	svc::Voxel v;
	v.Update( 0.005f, 0.02f, 2 );

	BOOST_REQUIRE( 2 == v.Weight() );
	BOOST_REQUIRE_CLOSE( 0.005f, v.Distance( 0.02f ), 0.1f );
}

BOOST_AUTO_TEST_CASE( UpdateWeight2 )
{
	svc::Voxel v1, v2;
	v1.Update( 0.005f, 0.02f );
	v2.Update( 0.005f, 0.02f, 1 );

	BOOST_REQUIRE( v1 == v2 );
}

BOOST_AUTO_TEST_CASE( equal )
{
	svc::Voxel v1;
	svc::Voxel v2;

	BOOST_REQUIRE( v1 == v1 );
	BOOST_REQUIRE( v1 == v2 );
	BOOST_REQUIRE( ! ( v1 != v1 ) );
	BOOST_REQUIRE( ! ( v1 != v2 ) );

	v1.Update( 0.1f, 0.271f );
	BOOST_REQUIRE( v1 == v1 );
	BOOST_REQUIRE( v1 != v2 );
	BOOST_REQUIRE( ! ( v1 == v2 ) );

	v2.Update( 0.1f, 0.271f );
	BOOST_REQUIRE( v1 == v2 );
	BOOST_REQUIRE( ! ( v1 != v2 ) );
}

BOOST_AUTO_TEST_SUITE_END()