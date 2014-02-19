#include "Volume.h"

#include <algorithm>
#include <cassert>
#include <cstdio>

#include "DepthFrame.h"
#include "flink.h"
#include "Timer.h"
#include "Voxel.h"

using namespace flink;



kppl::Volume::Volume( int resolution, float sideLength, float truncationMargin ) :
	m_res( resolution ),
	m_sideLen( sideLength ),
	m_truncationMargin( truncationMargin )
{
	assert( resolution > 0 );
	assert( sideLength > 0.0f );
	assert( truncationMargin > 0.0f );

	m_data.resize( resolution * resolution * resolution );
}



int kppl::Volume::Resolution() const
{
	return m_res;
}



kppl::Voxel const & kppl::Volume::operator()( int x, int y, int z ) const
{
	assert( IndicesAreValid( x, y, z ) );

	return m_data[ Index3Dto1D( x, y, z ) ];
}

kppl::Voxel & kppl::Volume::operator()( int x, int y, int z )
{
	assert( IndicesAreValid( x, y, z ) );

	return m_data[ Index3Dto1D( x, y, z ) ];
}



float4 kppl::Volume::VoxelCenter( int x, int y, int z ) const
{
	float4 result;

	int halfRes = Resolution() / 2;
	float voxelLen = m_sideLen / Resolution();

	result.x = ( x - halfRes + 0.5f ) * voxelLen;
	result.y = ( y - halfRes + 0.5f ) * voxelLen;
	result.z = ( z - halfRes + 0.5f ) * voxelLen;
	result.w = 1.0f;

	return result;
}



void kppl::Volume::Integrate
(
	kppl::DepthFrame const & frame, 
	float4 const & eye,
	float4 const & forward,
	float4x4 const & viewProjection
)
{
	assert( 0 == Resolution() % 2 );

	matrix _viewProj = load( & viewProjection );
	vector _ndcToUV = set( frame.Width() / 2.0f, frame.Height() / 2.0f, 0, 0 );

	for( int z = 0; z < Resolution(); z++ )
		for( int y = 0; y < Resolution(); y++ )
			for( int x = 0; x < Resolution(); x++ )
			{
				float4 centerWorld = VoxelCenter( x, y, z );
				vector _centerWorld = load( & centerWorld );

				vector _centerNDC = homogenize( _centerWorld * _viewProj );

				vector _centerScreen = _centerNDC * _ndcToUV + _ndcToUV;
				float4 centerScreen = store( _centerScreen );

				int u = (int) centerScreen.x;
				int v = (int) centerScreen.y;

				if( u < 0 || u >= frame.Width() || v < 0 || v >= frame.Height() )
					continue;

				float depth = frame( u, frame.Height() - v - 1 );

				if( depth == 0.0f )
					continue;

				float dist = dot( centerWorld - eye, forward );
				float signedDist = depth - dist;
				
				if( dist < 0.8f || signedDist < -m_truncationMargin )
					continue;

				(*this)( x, y, z ).Update( signedDist, m_truncationMargin );
			}
}



void kppl::Volume::Triangulate( char const * outOBJ )
{
#pragma region Type Defs

	struct Vertex
	{
		unsigned globalIdx;
		float x;
		float y;
		float z;

		Vertex() :
			globalIdx( 0 ), x( 0.0f ), y( 0.0f ), z( 0.0f )
		{
		}

		Vertex( int globalIdx, float x, float y, float z ) :
			globalIdx( globalIdx ), x( x ), y( y ), z( z )
		{
		}
	};

	struct VertexCmp
	{
		bool operator()( Vertex const & v1, Vertex const & v2 )
		{
			return v1.globalIdx < v2.globalIdx;
		}
	} vertexCmp;

#pragma endregion

#pragma region LUT

	static int const triTable[ 256 ][ 16 ] = {
		{ 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
		{ 3, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
		{ 3, 0, 1, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
		{ 6, 1, 8, 3, 9, 8, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
		{ 3, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
		{ 6, 0, 8, 3, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
		{ 6, 9, 2, 10, 0, 2, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
		{ 9, 2, 8, 3, 2, 10, 8, 10, 9, 8, -1, -1, -1, -1, -1, -1 },
		{ 3, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
		{ 6, 0, 11, 2, 8, 11, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
		{ 6, 1, 9, 0, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
		{ 9, 1, 11, 2, 1, 9, 11, 9, 8, 11, -1, -1, -1, -1, -1, -1 },
		{ 6, 3, 10, 1, 11, 10, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
		{ 9, 0, 10, 1, 0, 8, 10, 8, 11, 10, -1, -1, -1, -1, -1, -1 },
		{ 9, 3, 9, 0, 3, 11, 9, 11, 10, 9, -1, -1, -1, -1, -1, -1 },
		{ 6, 9, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
		{ 3, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
		{ 6, 4, 3, 0, 7, 3, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
		{ 6, 0, 1, 9, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
		{ 9, 4, 1, 9, 4, 7, 1, 7, 3, 1, -1, -1, -1, -1, -1, -1 },
		{ 6, 1, 2, 10, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
		{ 9, 3, 4, 7, 3, 0, 4, 1, 2, 10, -1, -1, -1, -1, -1, -1 },
		{ 9, 9, 2, 10, 9, 0, 2, 8, 4, 7, -1, -1, -1, -1, -1, -1 },
		{ 12, 2, 10, 9, 2, 9, 7, 2, 7, 3, 7, 9, 4, -1, -1, -1 },
		{ 6, 8, 4, 7, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
		{ 9, 11, 4, 7, 11, 2, 4, 2, 0, 4, -1, -1, -1, -1, -1, -1 },
		{ 9, 9, 0, 1, 8, 4, 7, 2, 3, 11, -1, -1, -1, -1, -1, -1 },
		{ 12, 4, 7, 11, 9, 4, 11, 9, 11, 2, 9, 2, 1, -1, -1, -1 },
		{ 9, 3, 10, 1, 3, 11, 10, 7, 8, 4, -1, -1, -1, -1, -1, -1 },
		{ 12, 1, 11, 10, 1, 4, 11, 1, 0, 4, 7, 11, 4, -1, -1, -1 },
		{ 12, 4, 7, 8, 9, 0, 11, 9, 11, 10, 11, 0, 3, -1, -1, -1 },
		{ 9, 4, 7, 11, 4, 11, 9, 9, 11, 10, -1, -1, -1, -1, -1, -1 },
		{ 3, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
		{ 6, 9, 5, 4, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
		{ 6, 0, 5, 4, 1, 5, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
		{ 9, 8, 5, 4, 8, 3, 5, 3, 1, 5, -1, -1, -1, -1, -1, -1 },
		{ 6, 1, 2, 10, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
		{ 9, 3, 0, 8, 1, 2, 10, 4, 9, 5, -1, -1, -1, -1, -1, -1 },
		{ 9, 5, 2, 10, 5, 4, 2, 4, 0, 2, -1, -1, -1, -1, -1, -1 },
		{ 12, 2, 10, 5, 3, 2, 5, 3, 5, 4, 3, 4, 8, -1, -1, -1 },
		{ 6, 9, 5, 4, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
		{ 9, 0, 11, 2, 0, 8, 11, 4, 9, 5, -1, -1, -1, -1, -1, -1 },
		{ 9, 0, 5, 4, 0, 1, 5, 2, 3, 11, -1, -1, -1, -1, -1, -1 },
		{ 12, 2, 1, 5, 2, 5, 8, 2, 8, 11, 4, 8, 5, -1, -1, -1 },
		{ 9, 10, 3, 11, 10, 1, 3, 9, 5, 4, -1, -1, -1, -1, -1, -1 },
		{ 12, 4, 9, 5, 0, 8, 1, 8, 10, 1, 8, 11, 10, -1, -1, -1 },
		{ 12, 5, 4, 0, 5, 0, 11, 5, 11, 10, 11, 0, 3, -1, -1, -1 },
		{ 9, 5, 4, 8, 5, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1 },
		{ 6, 9, 7, 8, 5, 7, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
		{ 9, 9, 3, 0, 9, 5, 3, 5, 7, 3, -1, -1, -1, -1, -1, -1 },
		{ 9, 0, 7, 8, 0, 1, 7, 1, 5, 7, -1, -1, -1, -1, -1, -1 },
		{ 6, 1, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
		{ 9, 9, 7, 8, 9, 5, 7, 10, 1, 2, -1, -1, -1, -1, -1, -1 },
		{ 12, 10, 1, 2, 9, 5, 0, 5, 3, 0, 5, 7, 3, -1, -1, -1 },
		{ 12, 8, 0, 2, 8, 2, 5, 8, 5, 7, 10, 5, 2, -1, -1, -1 },
		{ 9, 2, 10, 5, 2, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1 },
		{ 9, 7, 9, 5, 7, 8, 9, 3, 11, 2, -1, -1, -1, -1, -1, -1 },
		{ 12, 9, 5, 7, 9, 7, 2, 9, 2, 0, 2, 7, 11, -1, -1, -1 },
		{ 12, 2, 3, 11, 0, 1, 8, 1, 7, 8, 1, 5, 7, -1, -1, -1 },
		{ 9, 11, 2, 1, 11, 1, 7, 7, 1, 5, -1, -1, -1, -1, -1, -1 },
		{ 12, 9, 5, 8, 8, 5, 7, 10, 1, 3, 10, 3, 11, -1, -1, -1 },
		{ 15, 5, 7, 0, 5, 0, 9, 7, 11, 0, 1, 0, 10, 11, 10, 0 },
		{ 15, 11, 10, 0, 11, 0, 3, 10, 5, 0, 8, 0, 7, 5, 7, 0 },
		{ 6, 11, 10, 5, 7, 11, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
		{ 3, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
		{ 6, 0, 8, 3, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
		{ 6, 9, 0, 1, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
		{ 9, 1, 8, 3, 1, 9, 8, 5, 10, 6, -1, -1, -1, -1, -1, -1 },
		{ 6, 1, 6, 5, 2, 6, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
		{ 9, 1, 6, 5, 1, 2, 6, 3, 0, 8, -1, -1, -1, -1, -1, -1 },
		{ 9, 9, 6, 5, 9, 0, 6, 0, 2, 6, -1, -1, -1, -1, -1, -1 },
		{ 12, 5, 9, 8, 5, 8, 2, 5, 2, 6, 3, 2, 8, -1, -1, -1 },
		{ 6, 2, 3, 11, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
		{ 9, 11, 0, 8, 11, 2, 0, 10, 6, 5, -1, -1, -1, -1, -1, -1 },
		{ 9, 0, 1, 9, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1, -1, -1 },
		{ 12, 5, 10, 6, 1, 9, 2, 9, 11, 2, 9, 8, 11, -1, -1, -1 },
		{ 9, 6, 3, 11, 6, 5, 3, 5, 1, 3, -1, -1, -1, -1, -1, -1 },
		{ 12, 0, 8, 11, 0, 11, 5, 0, 5, 1, 5, 11, 6, -1, -1, -1 },
		{ 12, 3, 11, 6, 0, 3, 6, 0, 6, 5, 0, 5, 9, -1, -1, -1 },
		{ 9, 6, 5, 9, 6, 9, 11, 11, 9, 8, -1, -1, -1, -1, -1, -1 },
		{ 6, 5, 10, 6, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
		{ 9, 4, 3, 0, 4, 7, 3, 6, 5, 10, -1, -1, -1, -1, -1, -1 },
		{ 9, 1, 9, 0, 5, 10, 6, 8, 4, 7, -1, -1, -1, -1, -1, -1 },
		{ 12, 10, 6, 5, 1, 9, 7, 1, 7, 3, 7, 9, 4, -1, -1, -1 },
		{ 9, 6, 1, 2, 6, 5, 1, 4, 7, 8, -1, -1, -1, -1, -1, -1 },
		{ 12, 1, 2, 5, 5, 2, 6, 3, 0, 4, 3, 4, 7, -1, -1, -1 },
		{ 12, 8, 4, 7, 9, 0, 5, 0, 6, 5, 0, 2, 6, -1, -1, -1 },
		{ 15, 7, 3, 9, 7, 9, 4, 3, 2, 9, 5, 9, 6, 2, 6, 9 },
		{ 9, 3, 11, 2, 7, 8, 4, 10, 6, 5, -1, -1, -1, -1, -1, -1 },
		{ 12, 5, 10, 6, 4, 7, 2, 4, 2, 0, 2, 7, 11, -1, -1, -1 },
		{ 12, 0, 1, 9, 4, 7, 8, 2, 3, 11, 5, 10, 6, -1, -1, -1 },
		{ 15, 9, 2, 1, 9, 11, 2, 9, 4, 11, 7, 11, 4, 5, 10, 6 },
		{ 12, 8, 4, 7, 3, 11, 5, 3, 5, 1, 5, 11, 6, -1, -1, -1 },
		{ 15, 5, 1, 11, 5, 11, 6, 1, 0, 11, 7, 11, 4, 0, 4, 11 },
		{ 15, 0, 5, 9, 0, 6, 5, 0, 3, 6, 11, 6, 3, 8, 4, 7 },
		{ 12, 6, 5, 9, 6, 9, 11, 4, 7, 9, 7, 11, 9, -1, -1, -1 },
		{ 6, 10, 4, 9, 6, 4, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
		{ 9, 4, 10, 6, 4, 9, 10, 0, 8, 3, -1, -1, -1, -1, -1, -1 },
		{ 9, 10, 0, 1, 10, 6, 0, 6, 4, 0, -1, -1, -1, -1, -1, -1 },
		{ 12, 8, 3, 1, 8, 1, 6, 8, 6, 4, 6, 1, 10, -1, -1, -1 },
		{ 9, 1, 4, 9, 1, 2, 4, 2, 6, 4, -1, -1, -1, -1, -1, -1 },
		{ 12, 3, 0, 8, 1, 2, 9, 2, 4, 9, 2, 6, 4, -1, -1, -1 },
		{ 6, 0, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
		{ 9, 8, 3, 2, 8, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1 },
		{ 9, 10, 4, 9, 10, 6, 4, 11, 2, 3, -1, -1, -1, -1, -1, -1 },
		{ 12, 0, 8, 2, 2, 8, 11, 4, 9, 10, 4, 10, 6, -1, -1, -1 },
		{ 12, 3, 11, 2, 0, 1, 6, 0, 6, 4, 6, 1, 10, -1, -1, -1 },
		{ 15, 6, 4, 1, 6, 1, 10, 4, 8, 1, 2, 1, 11, 8, 11, 1 },
		{ 12, 9, 6, 4, 9, 3, 6, 9, 1, 3, 11, 6, 3, -1, -1, -1 },
		{ 15, 8, 11, 1, 8, 1, 0, 11, 6, 1, 9, 1, 4, 6, 4, 1 },
		{ 9, 3, 11, 6, 3, 6, 0, 0, 6, 4, -1, -1, -1, -1, -1, -1 },
		{ 6, 6, 4, 8, 11, 6, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
		{ 9, 7, 10, 6, 7, 8, 10, 8, 9, 10, -1, -1, -1, -1, -1, -1 },
		{ 12, 0, 7, 3, 0, 10, 7, 0, 9, 10, 6, 7, 10, -1, -1, -1 },
		{ 12, 10, 6, 7, 1, 10, 7, 1, 7, 8, 1, 8, 0, -1, -1, -1 },
		{ 9, 10, 6, 7, 10, 7, 1, 1, 7, 3, -1, -1, -1, -1, -1, -1 },
		{ 12, 1, 2, 6, 1, 6, 8, 1, 8, 9, 8, 6, 7, -1, -1, -1 },
		{ 15, 2, 6, 9, 2, 9, 1, 6, 7, 9, 0, 9, 3, 7, 3, 9 },
		{ 9, 7, 8, 0, 7, 0, 6, 6, 0, 2, -1, -1, -1, -1, -1, -1 },
		{ 6, 7, 3, 2, 6, 7, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
		{ 12, 2, 3, 11, 10, 6, 8, 10, 8, 9, 8, 6, 7, -1, -1, -1 },
		{ 15, 2, 0, 7, 2, 7, 11, 0, 9, 7, 6, 7, 10, 9, 10, 7 },
		{ 15, 1, 8, 0, 1, 7, 8, 1, 10, 7, 6, 7, 10, 2, 3, 11 },
		{ 12, 11, 2, 1, 11, 1, 7, 10, 6, 1, 6, 7, 1, -1, -1, -1 },
		{ 15, 8, 9, 6, 8, 6, 7, 9, 1, 6, 11, 6, 3, 1, 3, 6 },
		{ 6, 0, 9, 1, 11, 6, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
		{ 12, 7, 8, 0, 7, 0, 6, 3, 11, 0, 11, 6, 0, -1, -1, -1 },
		{ 3, 7, 11, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
		{ 3, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
		{ 6, 3, 0, 8, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
		{ 6, 0, 1, 9, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
		{ 9, 8, 1, 9, 8, 3, 1, 11, 7, 6, -1, -1, -1, -1, -1, -1 },
		{ 6, 10, 1, 2, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
		{ 9, 1, 2, 10, 3, 0, 8, 6, 11, 7, -1, -1, -1, -1, -1, -1 },
		{ 9, 2, 9, 0, 2, 10, 9, 6, 11, 7, -1, -1, -1, -1, -1, -1 },
		{ 12, 6, 11, 7, 2, 10, 3, 10, 8, 3, 10, 9, 8, -1, -1, -1 },
		{ 6, 7, 2, 3, 6, 2, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
		{ 9, 7, 0, 8, 7, 6, 0, 6, 2, 0, -1, -1, -1, -1, -1, -1 },
		{ 9, 2, 7, 6, 2, 3, 7, 0, 1, 9, -1, -1, -1, -1, -1, -1 },
		{ 12, 1, 6, 2, 1, 8, 6, 1, 9, 8, 8, 7, 6, -1, -1, -1 },
		{ 9, 10, 7, 6, 10, 1, 7, 1, 3, 7, -1, -1, -1, -1, -1, -1 },
		{ 12, 10, 7, 6, 1, 7, 10, 1, 8, 7, 1, 0, 8, -1, -1, -1 },
		{ 12, 0, 3, 7, 0, 7, 10, 0, 10, 9, 6, 10, 7, -1, -1, -1 },
		{ 9, 7, 6, 10, 7, 10, 8, 8, 10, 9, -1, -1, -1, -1, -1, -1 },
		{ 6, 6, 8, 4, 11, 8, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
		{ 9, 3, 6, 11, 3, 0, 6, 0, 4, 6, -1, -1, -1, -1, -1, -1 },
		{ 9, 8, 6, 11, 8, 4, 6, 9, 0, 1, -1, -1, -1, -1, -1, -1 },
		{ 12, 9, 4, 6, 9, 6, 3, 9, 3, 1, 11, 3, 6, -1, -1, -1 },
		{ 9, 6, 8, 4, 6, 11, 8, 2, 10, 1, -1, -1, -1, -1, -1, -1 },
		{ 12, 1, 2, 10, 3, 0, 11, 0, 6, 11, 0, 4, 6, -1, -1, -1 },
		{ 12, 4, 11, 8, 4, 6, 11, 0, 2, 9, 2, 10, 9, -1, -1, -1 },
		{ 15, 10, 9, 3, 10, 3, 2, 9, 4, 3, 11, 3, 6, 4, 6, 3 },
		{ 9, 8, 2, 3, 8, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1 },
		{ 6, 0, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
		{ 12, 1, 9, 0, 2, 3, 4, 2, 4, 6, 4, 3, 8, -1, -1, -1 },
		{ 9, 1, 9, 4, 1, 4, 2, 2, 4, 6, -1, -1, -1, -1, -1, -1 },
		{ 12, 8, 1, 3, 8, 6, 1, 8, 4, 6, 6, 10, 1, -1, -1, -1 },
		{ 9, 10, 1, 0, 10, 0, 6, 6, 0, 4, -1, -1, -1, -1, -1, -1 },
		{ 15, 4, 6, 3, 4, 3, 8, 6, 10, 3, 0, 3, 9, 10, 9, 3 },
		{ 6, 10, 9, 4, 6, 10, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
		{ 6, 4, 9, 5, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
		{ 9, 0, 8, 3, 4, 9, 5, 11, 7, 6, -1, -1, -1, -1, -1, -1 },
		{ 9, 5, 0, 1, 5, 4, 0, 7, 6, 11, -1, -1, -1, -1, -1, -1 },
		{ 12, 11, 7, 6, 8, 3, 4, 3, 5, 4, 3, 1, 5, -1, -1, -1 },
		{ 9, 9, 5, 4, 10, 1, 2, 7, 6, 11, -1, -1, -1, -1, -1, -1 },
		{ 12, 6, 11, 7, 1, 2, 10, 0, 8, 3, 4, 9, 5, -1, -1, -1 },
		{ 12, 7, 6, 11, 5, 4, 10, 4, 2, 10, 4, 0, 2, -1, -1, -1 },
		{ 15, 3, 4, 8, 3, 5, 4, 3, 2, 5, 10, 5, 2, 11, 7, 6 },
		{ 9, 7, 2, 3, 7, 6, 2, 5, 4, 9, -1, -1, -1, -1, -1, -1 },
		{ 12, 9, 5, 4, 0, 8, 6, 0, 6, 2, 6, 8, 7, -1, -1, -1 },
		{ 12, 3, 6, 2, 3, 7, 6, 1, 5, 0, 5, 4, 0, -1, -1, -1 },
		{ 15, 6, 2, 8, 6, 8, 7, 2, 1, 8, 4, 8, 5, 1, 5, 8 },
		{ 12, 9, 5, 4, 10, 1, 6, 1, 7, 6, 1, 3, 7, -1, -1, -1 },
		{ 15, 1, 6, 10, 1, 7, 6, 1, 0, 7, 8, 7, 0, 9, 5, 4 },
		{ 15, 4, 0, 10, 4, 10, 5, 0, 3, 10, 6, 10, 7, 3, 7, 10 },
		{ 12, 7, 6, 10, 7, 10, 8, 5, 4, 10, 4, 8, 10, -1, -1, -1 },
		{ 9, 6, 9, 5, 6, 11, 9, 11, 8, 9, -1, -1, -1, -1, -1, -1 },
		{ 12, 3, 6, 11, 0, 6, 3, 0, 5, 6, 0, 9, 5, -1, -1, -1 },
		{ 12, 0, 11, 8, 0, 5, 11, 0, 1, 5, 5, 6, 11, -1, -1, -1 },
		{ 9, 6, 11, 3, 6, 3, 5, 5, 3, 1, -1, -1, -1, -1, -1, -1 },
		{ 12, 1, 2, 10, 9, 5, 11, 9, 11, 8, 11, 5, 6, -1, -1, -1 },
		{ 15, 0, 11, 3, 0, 6, 11, 0, 9, 6, 5, 6, 9, 1, 2, 10 },
		{ 15, 11, 8, 5, 11, 5, 6, 8, 0, 5, 10, 5, 2, 0, 2, 5 },
		{ 12, 6, 11, 3, 6, 3, 5, 2, 10, 3, 10, 5, 3, -1, -1, -1 },
		{ 12, 5, 8, 9, 5, 2, 8, 5, 6, 2, 3, 8, 2, -1, -1, -1 },
		{ 9, 9, 5, 6, 9, 6, 0, 0, 6, 2, -1, -1, -1, -1, -1, -1 },
		{ 15, 1, 5, 8, 1, 8, 0, 5, 6, 8, 3, 8, 2, 6, 2, 8 },
		{ 6, 1, 5, 6, 2, 1, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
		{ 15, 1, 3, 6, 1, 6, 10, 3, 8, 6, 5, 6, 9, 8, 9, 6 },
		{ 12, 10, 1, 0, 10, 0, 6, 9, 5, 0, 5, 6, 0, -1, -1, -1 },
		{ 6, 0, 3, 8, 5, 6, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
		{ 3, 10, 5, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
		{ 6, 11, 5, 10, 7, 5, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
		{ 9, 11, 5, 10, 11, 7, 5, 8, 3, 0, -1, -1, -1, -1, -1, -1 },
		{ 9, 5, 11, 7, 5, 10, 11, 1, 9, 0, -1, -1, -1, -1, -1, -1 },
		{ 12, 10, 7, 5, 10, 11, 7, 9, 8, 1, 8, 3, 1, -1, -1, -1 },
		{ 9, 11, 1, 2, 11, 7, 1, 7, 5, 1, -1, -1, -1, -1, -1, -1 },
		{ 12, 0, 8, 3, 1, 2, 7, 1, 7, 5, 7, 2, 11, -1, -1, -1 },
		{ 12, 9, 7, 5, 9, 2, 7, 9, 0, 2, 2, 11, 7, -1, -1, -1 },
		{ 15, 7, 5, 2, 7, 2, 11, 5, 9, 2, 3, 2, 8, 9, 8, 2 },
		{ 9, 2, 5, 10, 2, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1 },
		{ 12, 8, 2, 0, 8, 5, 2, 8, 7, 5, 10, 2, 5, -1, -1, -1 },
		{ 12, 9, 0, 1, 5, 10, 3, 5, 3, 7, 3, 10, 2, -1, -1, -1 },
		{ 15, 9, 8, 2, 9, 2, 1, 8, 7, 2, 10, 2, 5, 7, 5, 2 },
		{ 6, 1, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
		{ 9, 0, 8, 7, 0, 7, 1, 1, 7, 5, -1, -1, -1, -1, -1, -1 },
		{ 9, 9, 0, 3, 9, 3, 5, 5, 3, 7, -1, -1, -1, -1, -1, -1 },
		{ 6, 9, 8, 7, 5, 9, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
		{ 9, 5, 8, 4, 5, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1 },
		{ 12, 5, 0, 4, 5, 11, 0, 5, 10, 11, 11, 3, 0, -1, -1, -1 },
		{ 12, 0, 1, 9, 8, 4, 10, 8, 10, 11, 10, 4, 5, -1, -1, -1 },
		{ 15, 10, 11, 4, 10, 4, 5, 11, 3, 4, 9, 4, 1, 3, 1, 4 },
		{ 12, 2, 5, 1, 2, 8, 5, 2, 11, 8, 4, 5, 8, -1, -1, -1 },
		{ 15, 0, 4, 11, 0, 11, 3, 4, 5, 11, 2, 11, 1, 5, 1, 11 },
		{ 15, 0, 2, 5, 0, 5, 9, 2, 11, 5, 4, 5, 8, 11, 8, 5 },
		{ 6, 9, 4, 5, 2, 11, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
		{ 12, 2, 5, 10, 3, 5, 2, 3, 4, 5, 3, 8, 4, -1, -1, -1 },
		{ 9, 5, 10, 2, 5, 2, 4, 4, 2, 0, -1, -1, -1, -1, -1, -1 },
		{ 15, 3, 10, 2, 3, 5, 10, 3, 8, 5, 4, 5, 8, 0, 1, 9 },
		{ 12, 5, 10, 2, 5, 2, 4, 1, 9, 2, 9, 4, 2, -1, -1, -1 },
		{ 9, 8, 4, 5, 8, 5, 3, 3, 5, 1, -1, -1, -1, -1, -1, -1 },
		{ 6, 0, 4, 5, 1, 0, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
		{ 12, 8, 4, 5, 8, 5, 3, 9, 0, 5, 0, 3, 5, -1, -1, -1 },
		{ 3, 9, 4, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
		{ 9, 4, 11, 7, 4, 9, 11, 9, 10, 11, -1, -1, -1, -1, -1, -1 },
		{ 12, 0, 8, 3, 4, 9, 7, 9, 11, 7, 9, 10, 11, -1, -1, -1 },
		{ 12, 1, 10, 11, 1, 11, 4, 1, 4, 0, 7, 4, 11, -1, -1, -1 },
		{ 15, 3, 1, 4, 3, 4, 8, 1, 10, 4, 7, 4, 11, 10, 11, 4 },
		{ 12, 4, 11, 7, 9, 11, 4, 9, 2, 11, 9, 1, 2, -1, -1, -1 },
		{ 15, 9, 7, 4, 9, 11, 7, 9, 1, 11, 2, 11, 1, 0, 8, 3 },
		{ 9, 11, 7, 4, 11, 4, 2, 2, 4, 0, -1, -1, -1, -1, -1, -1 },
		{ 12, 11, 7, 4, 11, 4, 2, 8, 3, 4, 3, 2, 4, -1, -1, -1 },
		{ 12, 2, 9, 10, 2, 7, 9, 2, 3, 7, 7, 4, 9, -1, -1, -1 },
		{ 15, 9, 10, 7, 9, 7, 4, 10, 2, 7, 8, 7, 0, 2, 0, 7 },
		{ 15, 3, 7, 10, 3, 10, 2, 7, 4, 10, 1, 10, 0, 4, 0, 10 },
		{ 6, 1, 10, 2, 8, 7, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
		{ 9, 4, 9, 1, 4, 1, 7, 7, 1, 3, -1, -1, -1, -1, -1, -1 },
		{ 12, 4, 9, 1, 4, 1, 7, 0, 8, 1, 8, 7, 1, -1, -1, -1 },
		{ 6, 4, 0, 3, 7, 4, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
		{ 3, 4, 8, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
		{ 6, 9, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
		{ 9, 3, 0, 9, 3, 9, 11, 11, 9, 10, -1, -1, -1, -1, -1, -1 },
		{ 9, 0, 1, 10, 0, 10, 8, 8, 10, 11, -1, -1, -1, -1, -1, -1 },
		{ 6, 3, 1, 10, 11, 3, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
		{ 9, 1, 2, 11, 1, 11, 9, 9, 11, 8, -1, -1, -1, -1, -1, -1 },
		{ 12, 3, 0, 9, 3, 9, 11, 1, 2, 9, 2, 11, 9, -1, -1, -1 },
		{ 6, 0, 2, 11, 8, 0, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
		{ 3, 3, 2, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
		{ 9, 2, 3, 8, 2, 8, 10, 10, 8, 9, -1, -1, -1, -1, -1, -1 },
		{ 6, 9, 10, 2, 0, 9, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
		{ 12, 2, 3, 8, 2, 8, 10, 0, 1, 8, 1, 10, 8, -1, -1, -1 },
		{ 3, 1, 10, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
		{ 6, 1, 3, 8, 9, 1, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
		{ 3, 0, 9, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
		{ 3, 0, 3, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
		{ 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 }
	};

#pragma endregion

	std::vector< Vertex > VB;
	std::vector< unsigned > IB;

	Timer timer;

	int resMinus1 = Resolution() - 1;
	float vLen = m_sideLen / Resolution();

	for( int z0 = 0; z0 < Resolution(); z0++ )
		for( int y0 = 0; y0 < Resolution(); y0++ )
			for( int x0 = 0; x0 < Resolution(); x0++ )
			{
				int x1 = std::min( x0 + 1, resMinus1 );
				int y1 = std::min( y0 + 1, resMinus1 );
				int z1 = std::min( z0 + 1, resMinus1 );

				Voxel v[ 8 ];
				v[ 2 ] = (*this)( x0, y0, z0 );
				v[ 3 ] = (*this)( x1, y0, z0 );
				v[ 6 ] = (*this)( x0, y1, z0 );
				v[ 7 ] = (*this)( x1, y1, z0 );

				v[ 1 ] = (*this)( x0, y0, z1 );
				v[ 0 ] = (*this)( x1, y0, z1 );
				v[ 5 ] = (*this)( x0, y1, z1 );
				v[ 4 ] = (*this)( x1, y1, z1 );

				// Generate vertices
				if( 0 == v[ 2 ].Weight() )
					continue;

				float d[ 8 ];
				d[ 1 ] = v[ 1 ].Distance( m_truncationMargin );
				d[ 2 ] = v[ 2 ].Distance( m_truncationMargin );
				d[ 3 ] = v[ 3 ].Distance( m_truncationMargin );
				d[ 6 ] = v[ 6 ].Distance( m_truncationMargin );

				float4 vert000 = VoxelCenter( x0, y0, z0 );
				unsigned i000 = Index3Dto1D( x0, y0, z0 );

				if( v[ 3 ].Weight() > 0 && d[ 2 ] * d[ 3 ] < 0.0f )
					VB.push_back( Vertex
					(
						3 * i000,
						vert000.x + lerp( 0.0f, vLen, v[ 2 ].Weight() * abs( d[ 3 ] ), v[ 3 ].Weight() * abs( d[ 2 ] ) ),
						vert000.y,
						vert000.z
					));
				
				if( v[ 6 ].Weight() > 0 && d[ 2 ] * d[ 6 ] < 0.0f )
					VB.push_back( Vertex
					(
						3 * i000 + 1,
						vert000.x,
						vert000.y + lerp( 0.0f, vLen, v[ 2 ].Weight() * abs( d[ 6 ] ), v[ 6 ].Weight() * abs( d[ 2 ] ) ),
						vert000.z
					));
				
				if( v[ 1 ].Weight() > 0 && d[ 2 ] * d[ 1 ] < 0.0f )
					VB.push_back( Vertex
					(
						3 * i000 + 2,
						vert000.x,
						vert000.y,
						vert000.z + lerp( 0.0f, vLen, v[ 2 ].Weight() * abs( d[ 1 ] ), v[ 1 ].Weight() * abs( d[ 2 ] ) )
					));

				// Generate indices
				bool skip = false;
				for( int i = 0; i < 8; i++ )
					skip |= ( 0 == v[ i ].Weight() );

				if( skip ||					
					x0 == resMinus1 ||
					y0 == resMinus1 ||
					z0 == resMinus1 )
					continue;

				d[ 0 ] = v[ 0 ].Distance( m_truncationMargin );
				d[ 4 ] = v[ 4 ].Distance( m_truncationMargin );
				d[ 5 ] = v[ 5 ].Distance( m_truncationMargin );
				d[ 7 ] = v[ 7 ].Distance( m_truncationMargin );

				int lutIdx = 0;
				for( int i = 0; i < 8; i++ )
					if( d[ i ] < 0 )
						lutIdx |= ( 1u << i );

				// Maps local edge indices to global vertex indices
				unsigned localToGlobal[ 12 ];
				localToGlobal[  0 ] = Index3Dto1D( x0, y0, z1 ) * 3;
				localToGlobal[  1 ] = Index3Dto1D( x0, y0, z0 ) * 3 + 2;
				localToGlobal[  2 ] = Index3Dto1D( x0, y0, z0 ) * 3;
				localToGlobal[  3 ] = Index3Dto1D( x1, y0, z0 ) * 3 + 2;
				localToGlobal[  4 ] = Index3Dto1D( x0, y1, z1 ) * 3;
				localToGlobal[  5 ] = Index3Dto1D( x0, y1, z0 ) * 3 + 2;
				localToGlobal[  6 ] = Index3Dto1D( x0, y1, z0 ) * 3;
				localToGlobal[  7 ] = Index3Dto1D( x1, y1, z0 ) * 3 + 2;
				localToGlobal[  8 ] = Index3Dto1D( x1, y0, z1 ) * 3 + 1;
				localToGlobal[  9 ] = Index3Dto1D( x0, y0, z1 ) * 3 + 1;
				localToGlobal[ 10 ] = Index3Dto1D( x0, y0, z0 ) * 3 + 1;
				localToGlobal[ 11 ] = Index3Dto1D( x1, y0, z0 ) * 3 + 1;

				for( int i = 0; i < triTable[ lutIdx ][ 0 ]; i++ )
					IB.push_back( localToGlobal[ triTable[ lutIdx ][ i + 1 ] ] );
			}

	std::sort( VB.begin(), VB.end(), vertexCmp );

	Vertex dummy;
	for( int i = 0; i < IB.size(); i++ )
	{
		dummy.globalIdx = IB[ i ];
		auto it = std::lower_bound( VB.cbegin(), VB.cend(), dummy, vertexCmp );
		IB[ i ] = (unsigned) ( it - VB.cbegin() );
	}

	printf( "MC, gen vb+ib: %fs\n", timer.Time() );
	timer.Reset();

	// TODO: Remove unused vertices from VB
	// or test how high their percentage is and possibly leave them in.

	FILE * file;
	fopen_s( & file, outOBJ, "w" );

	for( int i = 0; i < VB.size(); i++ )
		fprintf_s( file, "v %f %f %f\n", VB[ i ].x, VB[ i ].y, VB[ i ].z );

	for( int i = 0; i < IB.size(); i += 3 )
		fprintf_s( file, "f %d %d %d\n", IB[ i ] + 1, IB[ i + 1 ] + 1, IB[ i + 2 ] + 1 );

	fclose( file );
}



bool kppl::Volume::IndicesAreValid( int x, int y, int z ) const
{
	return
		x >= 0 &&
		y >= 0 &&
		z >= 0 &&

		x < Resolution() &&
		y < Resolution() &&
		z < Resolution();
}

unsigned kppl::Volume::Index3Dto1D( unsigned x, unsigned y, unsigned z ) const
{
	assert( IndicesAreValid( x, y, z ) );
	assert( Resolution() <= 1024 );

	return ( z * Resolution() + y ) * Resolution() + x;
}