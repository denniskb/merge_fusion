#include "Volume.h"

#include <algorithm>
#include <cassert>
#include <cstdio>

#include <flink/algorithm.h>
#include <flink/math.h>
#include <flink/util.h>

#include "DepthFrame.h"
#include "Voxel.h"



template< int BrickRes >
svc::Volume< BrickRes >::Volume( int resolution, float sideLength, float truncMargin ) :
	m_res( resolution ),
	m_sideLen( sideLength ),
	m_truncMargin( truncMargin )
{
	static_assert( 
		BrickRes == 1 || BrickRes == 2 || BrickRes == 4,
		"BrickRes \\in { 1, 2, 4 }"
	);

	assert( resolution > 0 && resolution <= 1024 );
	assert( sideLength > 0.0f );
	assert( truncMargin > 0.0f );

	assert( flink::powerOf2( resolution ) );
}



template< int BrickRes >
int svc::Volume< BrickRes >::Resolution() const
{
	return m_res;
}

template< int BrickRes >
float svc::Volume< BrickRes >::SideLength() const
{
	return m_sideLen;
}

template< int BrickRes >
float svc::Volume< BrickRes >::TruncationMargin() const
{
	return m_truncMargin;
}



template< int BrickRes >
float svc::Volume< BrickRes >::VoxelLength() const
{
	return m_sideLen / m_res;
}

template< int BrickRes >
int svc::Volume< BrickRes >::BrickSlice() const
{
	return BrickRes * BrickRes;
}

template< int BrickRes >
int svc::Volume< BrickRes >::BrickVolume() const
{
	return BrickRes * BrickRes * BrickRes;
}

template< int BrickRes >
int svc::Volume< BrickRes >::NumBricksInVolume() const
{
	return m_res / BrickRes;
}



template< int BrickRes >
flink::float4 svc::Volume< BrickRes >::Minimum() const
{
	float minimum = -m_sideLen * 0.5f;

	return flink::float4
	(
		minimum,
		minimum,
		minimum,
		1.0f
	);
}

template< int BrickRes >
flink::float4 svc::Volume< BrickRes >::Maximum() const
{
	float maximum = 0.5f * m_sideLen;

	return flink::float4
	(
		maximum,
		maximum,
		maximum,
		1.0f
	);
}



template< int BrickRes >
flink::float4 svc::Volume< BrickRes >::VoxelCenter( int x, int y, int z ) const
{
	assert( x >= 0 && x < Resolution() );
	assert( y >= 0 && y < Resolution() );
	assert( z >= 0 && z < Resolution() );

	return 
		Minimum() +
		flink::float4
		( 
			( x + 0.5f ) / Resolution(), 
			( y + 0.5f ) / Resolution(), 
			( z + 0.5f ) / Resolution(), 
			1.0f
		) *
		( Maximum() - Minimum() );
}

template< int BrickRes >
flink::float4 svc::Volume< BrickRes >::BrickIndex( flink::float4 const & world ) const
{
	return
		( world - Minimum() ) / ( Maximum() - Minimum() ) *
		flink::make_float4( (float) NumBricksInVolume() );
}



template< int BrickRes >
flink::flat_map< unsigned, svc::Brick< BrickRes > > & svc::Volume< BrickRes >::Data()
{
	return m_data;
}

template< int BrickRes >
flink::flat_map< unsigned, svc::Brick< BrickRes > > const & svc::Volume< BrickRes >::Data() const
{
	return m_data;
}



template svc::Volume< 1 >;
template svc::Volume< 2 >;
template svc::Volume< 4 >;