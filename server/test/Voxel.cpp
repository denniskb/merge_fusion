#include <boost/test/auto_unit_test.hpp>

#include <server/Voxel.m>



BOOST_AUTO_TEST_SUITE( Voxel )

BOOST_AUTO_TEST_CASE( Update )
{
	float const tm = 0.271f;

	kppl::Voxel v;
	BOOST_REQUIRE( 0 == v.Weight() );

	v.Update( 0.1, tm );
	BOOST_REQUIRE_CLOSE( v.Distance( tm ), 0.1f, 0.1f );
	BOOST_REQUIRE( 1 == v.Weight() );

	kppl::Voxel v2;
	v2.Update( 0.5f, tm );
	BOOST_REQUIRE_CLOSE( v2.Distance( tm ), tm, 0.1f );
}

BOOST_AUTO_TEST_CASE( equal )
{
	kppl::Voxel v1;
	kppl::Voxel v2;

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

BOOST_AUTO_TEST_CASE( shortcut )
{
	kppl::Voxel v1;
	kppl::Voxel v2;

	v1.Update( 0.01f, 0.02f );
	v2.Update( 0.01f, 0.02f, kppl::Voxel::PACK_SCALE( 0.02f ), kppl::Voxel::UNPACK_SCALE( 0.02f ) );

	BOOST_REQUIRE( v1 == v2 );
	BOOST_REQUIRE( v1.Distance( 0.02f ) == v2.Distance( 0.02f, kppl::Voxel::UNPACK_SCALE( 0.02f ) ) );
}

BOOST_AUTO_TEST_SUITE_END()