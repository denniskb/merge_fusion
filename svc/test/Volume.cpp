#include <boost/test/auto_unit_test.hpp>

#include <reference/Volume.h>



BOOST_AUTO_TEST_SUITE( Volume )

BOOST_AUTO_TEST_CASE( ctor )
{
	svc::Volume v( 128, 2.0f, 0.02f );

	BOOST_REQUIRE( v.Resolution() == 128 );
	BOOST_REQUIRE( v.SideLength() == 2.0f );
	BOOST_REQUIRE( v.TruncationMargin() == 0.02f );
	BOOST_REQUIRE( v.VoxelLength() == 2.0f / 128.0f );

	BOOST_REQUIRE( v.NumChunksInVolume( 1 ) == 128 );
	BOOST_REQUIRE( v.NumChunksInVolume( 2 ) ==  64 );
	BOOST_REQUIRE( v.NumChunksInVolume( 4 ) ==  32 );

	BOOST_REQUIRE( v.Minimum().x == -1.0f );
	BOOST_REQUIRE( v.Maximum().y == 1.0f );
}

BOOST_AUTO_TEST_CASE( VoxelCenter_ChunkIndex )
{
	svc::Volume v( 128, 2.0f, 0.02f );
	
	flink::float4 vc = v.VoxelCenter( 33, 21, 92 );
	flink::float4 ci = v.ChunkIndex( vc, 1 );

	BOOST_REQUIRE_CLOSE( 33.5f, ci.x, 0.1f );
	BOOST_REQUIRE_CLOSE( 21.5f, ci.y, 0.1f );
	BOOST_REQUIRE_CLOSE( 92.5f, ci.z, 0.1f );

	// TODO: Test ChunkIndex != 1
}

BOOST_AUTO_TEST_SUITE_END()