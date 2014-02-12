#include "Volume.h"

#include <cassert>

#include "flink.h"
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
	std::vector< short > const & frame, 
	float4x4 const & view,
	float4x4 const & projection
)
{
	assert( 0 == Resolution() % 2 );

	matrix _view = load( & view );
	matrix _projection = load( & projection );
	vector _ndcToUV = set( 320, 240, 0, 0 );

	for( int z = 0; z < Resolution(); z++ )
		for( int y = 0; y < Resolution(); y++ )
			for( int x = 0; x < Resolution(); x ++ )
			{
				float4 centerWorld = VoxelCenter( x, y, z );
				vector _centerWorld = load( & centerWorld );

				vector _centerCamera = _centerWorld * _view;
				float4 centerCamera = store( _centerCamera );

				vector _centerNDC = homogenize( _centerCamera * _projection );
				vector _centerScreen = _centerNDC * _ndcToUV + _ndcToUV;

				float4 centerScreen = store( _centerScreen );

				int u = (int) centerScreen.x;
				int v = (int) centerScreen.y;

				if( u < 0 || u > 639 || v < 0 || v > 479 )
					continue;

				int txIdx = u + 640 * ( 479 - v );
				float depth = frame[ txIdx ] * 0.001f;

				if( depth == 0 )
					continue;

				float dist = -centerCamera.z;
				float signedDist = depth - dist;
				
				if( dist < 0.4f || signedDist < -m_truncationMargin )
					continue;

				(*this)( x, y, z ).Update( signedDist, m_truncationMargin );
			}
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

int kppl::Volume::Index3Dto1D( int x, int y, int z ) const
{
	assert( IndicesAreValid( x, y, z ) );
	assert( Resolution() <= 1024 );

	return ( z * Resolution() + y ) * Resolution() + x;
}