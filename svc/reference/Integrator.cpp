#include "Integrator.h"

#include <flink/algorithm.h>
#include <flink/util.h>
#include <flink/vector.h>

#include "Cache.h"
#include "DepthFrame.h"
#include "Volume.h"
#include "Voxel.h"



// static 
void svc::Integrator::Integrate
( 
	Volume & volume,
	Cache & cache,
	DepthFrame const & frame,

	flink::float4 const & eye,
	flink::float4 const & forward,

	flink::float4x4 const & viewProjection,
	flink::float4x4 const & viewToWorld
)
{
	SplatBricks( volume, frame, viewToWorld, m_splattedVoxels );
	radix_sort( m_splattedVoxels );
	remove_dups( m_splattedVoxels );

	ExpandBricks( volume, cache, m_splattedVoxels );
	
	BricksToVoxels( volume, m_splattedVoxels );
	radix_sort( m_splattedVoxels );

	volume.Indices() = m_splattedVoxels;
	volume.Voxels().resize( volume.Indices().size() );

	UpdateVoxels( volume, frame, eye, forward, viewProjection );
}



// static 
void svc::Integrator::SplatBricks
(
	Volume const & volume,
	DepthFrame const & frame,
	flink::float4x4 const & viewToWorld,

	flink::vector< unsigned > & outBrickIndices
)
{
	outBrickIndices.clear();

	flink::mat _viewToWorld = flink::load( viewToWorld );

	float const halfFrameWidth = (float) ( frame.Width() / 2 );
	float const halfFrameHeight = (float) ( frame.Height() / 2 );

	float const ppX = halfFrameWidth - 0.5f;
	float const ppY = halfFrameHeight - 0.5f;

	// TODO: Encapsulate in CameraParams struct!
	float const fl = 585.0f;

	for( int i = 0, res = frame.Resolution(); i < res; i++ )
	{
		int y = i / frame.Width();
		int x = i % frame.Width();

		float depth = frame( x, y );
		if( 0.0f == depth )
			continue;

		float xNdc = ( x - ppX ) / halfFrameWidth;
		float yNdc = ( ppY - y ) / halfFrameHeight;

		flink::float4 pxView
		(
			xNdc * ( halfFrameWidth / fl ) * depth,
			yNdc * ( halfFrameHeight / fl ) * depth,
			-depth,
			1.0f
		);

		flink::vec _pxView = flink::load( pxView );
		flink::vec _pxWorld = _pxView * _viewToWorld;

		flink::float4 pxWorld = flink::store( _pxWorld );
		flink::float4 pxVol = volume.BrickIndex( pxWorld );

		if( pxVol < flink::make_float4( 0.5f ) ||
			pxVol >= flink::make_float4( volume.NumBricksInVolume() - 0.5f ) )
			continue;

		outBrickIndices.push_back( flink::packInts
		(
			(unsigned) ( pxVol.x - 0.5f ),
			(unsigned) ( pxVol.y - 0.5f ),
			(unsigned) ( pxVol.z - 0.5f )
		));
	}
}

// static 
void svc::Integrator::ExpandBricks
(
	Volume const & volume,
	Cache & cache,

	flink::vector< unsigned > & inOutBrickIndices
)
{
	ExpandBricksHelper< 1 >( volume, cache, 0, flink::packInts( 0, 0, 1 ), inOutBrickIndices );
	ExpandBricksHelper< 0 >( volume, cache, volume.NumBricksInVolume(), flink::packInts( 0, 1, 0 ), inOutBrickIndices );
	ExpandBricksHelperX( volume.NumBricksInVolume(), inOutBrickIndices );
}

// static
template< int sliceIdx >
void svc::Integrator::ExpandBricksHelper
(
	Volume const & volume,
	Cache & cache,
	int deltaLookUp,
	unsigned deltaStore,

	flink::vector< unsigned > & inOutBrickIndices
)
{
	// This method can only be used to expand in z or y direction
	// otherwise invalid voxel indices are generated
	assert( deltaLookUp == 0 || deltaLookUp == volume.NumBricksInVolume() );

	int size = inOutBrickIndices.size();
	cache.Reset( volume.NumBricksInVolume() );

	while( cache.NextSlice( inOutBrickIndices.cbegin(), inOutBrickIndices.cbegin(), size ) )
	{
		for( int i = cache.CachedRange().first; i < cache.CachedRange().second; i++ )
		{
			int idx = 
				flink::unpackX( inOutBrickIndices[ i ] ) + 
				flink::unpackY( inOutBrickIndices[ i ] ) * cache.SliceRes() + 
				deltaLookUp;
			
			if( idx < cache.SliceSize() &&
				std::get< sliceIdx >( cache.CachedSlices() )[ idx ] == 0 )
				inOutBrickIndices.push_back( inOutBrickIndices[ i ] + deltaStore );
		}
	}
	flink::radix_sort( inOutBrickIndices );
}

template void svc::Integrator::ExpandBricksHelper< 0 >(Volume const &, Cache &, int, unsigned, flink::vector< unsigned > &);
template void svc::Integrator::ExpandBricksHelper< 1 >(Volume const &, Cache &, int, unsigned, flink::vector< unsigned > &);

// static 
void svc::Integrator::ExpandBricksHelperX
(
	int numBricksInVolume,
	flink::vector< unsigned > & inOutBrickIndices
)
{
	int size = inOutBrickIndices.size();
	inOutBrickIndices.resize( size * 2 );
	std::memset( inOutBrickIndices.begin() + size, 0, size * sizeof( unsigned ) );
	
	unsigned tmp = 0;
	for( int i = size - 1; i >= 0; i-- )
	{
		unsigned xyz = inOutBrickIndices[ i ];
		inOutBrickIndices[ i ] = 0;
		
		inOutBrickIndices[ 2 * i ] = xyz;
		
		if( (int) flink::unpackX( xyz ) + 1 < numBricksInVolume &&
			tmp != xyz + 1 )
			inOutBrickIndices[ 2 * i + 1 ] = xyz + 1;
	
		tmp = xyz;
	}
		
	remove_value( inOutBrickIndices, 0 );
}

// static 
void svc::Integrator::BricksToVoxels
(
	Volume const & volume,
	flink::vector< unsigned > & inOutIndices
)
{
	if( volume.BrickResolution() > 1 )
	{
		int const brickSlice = volume.BrickSlice();
		int const brickVolume = volume.BrickVolume();

		int const size = inOutIndices.size();
		inOutIndices.resize( size * brickVolume );

		for( int i = size - 1; i >= 0; i-- )
		{
			unsigned brickX, brickY, brickZ;
			flink::unpackInts( inOutIndices[ i ], brickX, brickY, brickZ );

			for( int j = 0; j < brickVolume; j++ )
			{
				unsigned z = j / brickSlice;
				unsigned y = ( j - z * brickSlice ) / volume.BrickResolution();
				unsigned x = j % volume.BrickResolution();

				inOutIndices[ i * brickVolume + j ] = flink::packInts
				(
					brickX * volume.BrickResolution() + x,
					brickY * volume.BrickResolution() + y,
					brickZ * volume.BrickResolution() + z
				);
			}
		}
	}
}

// static 
void svc::Integrator::UpdateVoxels
(
	Volume & volume,

	DepthFrame const & frame, 

	flink::float4 const & eye,
	flink::float4 const & forward,
	flink::float4x4 const & viewProjection
)
{
	flink::mat _viewProj = flink::load( viewProjection );
	flink::vec _ndcToUV = flink::set( frame.Width() / 2.0f, frame.Height() / 2.0f, 0, 0 );
		
	for( int i = 0; i < volume.Indices().size(); i++ )
	{
		unsigned x, y, z;
		flink::unpackInts( volume.Indices()[ i ], x, y, z );

		flink::float4 centerWorld = volume.VoxelCenter( x, y, z );
		flink::vec _centerWorld = flink::load( centerWorld );
		
		flink::vec _centerNDC = flink::homogenize( _centerWorld * _viewProj );
		
		flink::vec _centerScreen = _mm_macc_ps( _centerNDC, _ndcToUV, _ndcToUV );
		flink::float4 centerScreen = flink::store( _centerScreen );

		int u = (int) centerScreen.x;
		int v = (int) centerScreen.y;

		float depth = 0.0f;
		// CUDA: Clamp out of bounds access to 0 to avoid divergence
		if( u >= 0 && u < frame.Width() && v >= 0 && v < frame.Height() )
			depth = frame( u, frame.Height() - v - 1 );

		bool update = ( depth != 0.0f );

		float dist = flink::dot( centerWorld - eye, forward );
		float signedDist = depth - dist;
				
		update = update && ( dist >= 0.8f && signedDist >= -volume.TruncationMargin() );

		Voxel vx;
		vx.Update( signedDist, volume.TruncationMargin(), (int) update );
		volume.Voxels()[ i ] = vx;
	}
}