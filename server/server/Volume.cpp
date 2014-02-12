#include "Volume.h"

#include <cassert>

#include <DirectXMath.h>

#include "Voxel.h"

using namespace DirectX;



kppl::Volume::Volume( int resolution, float sideLength ) :
	m_res( resolution ),
	m_sideLen( sideLength )
{
	assert( resolution > 0 );
	assert( sideLength > 0.0f );

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



XMFLOAT4A kppl::Volume::VoxelCenter( int x, int y, int z ) const
{
	XMFLOAT4A result;

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
	XMFLOAT4X4A const & view,
	XMFLOAT4X4A const & projection,
	float truncationMargin
)
{
	assert( truncationMargin > 0 );
	assert( 0 == Resolution() % 2 );

	XMMATRIX _view = XMLoadFloat4x4A( & view );
	XMMATRIX _projection = XMLoadFloat4x4A( & projection );
	XMVECTOR _ndcToUV = XMVectorSet( 320, 240, 0, 0 );

	for( int z = 0; z < Resolution(); z++ )
		for( int y = 0; y < Resolution(); y++ )
			for( int x = 0; x < Resolution(); x ++ )
			{
				XMFLOAT4A centerWorld = VoxelCenter( x, y, z );
				XMVECTOR _centerWorld = XMLoadFloat4A( & centerWorld );

				XMVECTOR _centerCamera = XMVector4Transform( _centerWorld, _view );
				XMFLOAT4A centerCamera; XMStoreFloat4A( & centerCamera, _centerCamera );

				XMVECTOR _centerScreen = XMVector4Transform( _centerCamera, _projection );
				XMVECTOR _wwww = XMVectorPermute< 3, 3, 3, 3 >( _centerScreen, _centerScreen );
				XMVECTOR _centerNDC = XMVectorDivide( _centerScreen, _wwww );
				XMVECTOR _centerUV = _centerNDC * _ndcToUV + _ndcToUV;

				XMFLOAT4A centerUV;
				XMStoreFloat4A( & centerUV, _centerUV );

				int u = (int) centerUV.x;
				int v = (int) centerUV.y;

				if( u < 0 || u > 639 || v < 0 || v > 479 )
					continue;

				int txIdx = u + 640 * ( 479 - v );
				float depth = frame[ txIdx ] * 0.001f;

				if( depth == 0 )
					continue;

				float dist = -centerCamera.z;
				float signedDist = depth - dist;
				
				if( dist < 0.4f || signedDist < -truncationMargin )
					continue;

				(*this)( x, y, z ).Update( signedDist, truncationMargin );
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