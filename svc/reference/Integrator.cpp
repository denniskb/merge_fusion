#include "Integrator.h"

#include <flink/algorithm.h>
#include <flink/util.h>
#include <flink/vector.h>

#include "Cache.h"
#include "DepthFrame.h"
#include "Volume.h"
#include "Voxel.h"

#include <flink/timer.h>



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
	double tsplat, tsort, tdups, texpand, tbricks, tsort2, tmerge, tupdate;
	flink::timer t;
	SplatBricks( volume, frame, viewToWorld, m_splattedVoxels );
	tsplat = t.time(); t.reset();
	flink::radix_sort( m_splattedVoxels.begin(), m_splattedVoxels.size(), m_scratchPad );
	tsort = t.time(); t.reset();
	remove_dups( m_splattedVoxels );
	tdups = t.time(); t.reset();

	ExpandBricks( volume, cache, m_splattedVoxels, m_scratchPad );
	texpand = t.time(); t.reset();
	
	BricksToVoxels( volume, m_splattedVoxels );
	tbricks = t.time(); t.reset();
	flink::radix_sort( m_splattedVoxels.begin(), m_splattedVoxels.size(), m_scratchPad );
	tsort2 = t.time(); t.reset();

	volume.Data().merge_unique( m_splattedVoxels.cbegin(), m_splattedVoxels.cend(), 0 );
	tmerge = t.time(); t.reset();

	UpdateVoxels( volume, frame, eye, forward, viewProjection );
	tupdate = t.time();

	printf( "tsplat: %fms\n", tsplat * 1000.0 );
	printf( "tsort: %fms\n", tsort * 1000.0 );
	printf( "tdups: %fms\n", tdups * 1000.0 );
	printf( "texpand: %fms\n", texpand * 1000.0 );

	printf( "tbricks: %fms\n", tbricks * 1000.0 );
	printf( "tsort2: %fms\n", tsort2 * 1000.0 );
	printf( "tmerge: %fms\n", tmerge * 1000.0 );
	printf( "tupdate: %fms\n", tupdate * 1000.0 );

	printf( "ttotal: %fms\n\n", (tsplat+tsort+tdups+texpand+tbricks+tsort2+tmerge+tupdate)*1000.0 );
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

	flink::vector< unsigned > & inOutBrickIndices,
	flink::vector< char > & scratchPad
)
{
	ExpandBricksHelper< 1 >( volume, cache, 0, flink::packZ( 1 ), inOutBrickIndices, scratchPad );
	ExpandBricksHelper< 0 >( volume, cache, volume.NumBricksInVolume(), flink::packY( 1 ), inOutBrickIndices, scratchPad );
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

	flink::vector< unsigned > & inOutBrickIndices,
	flink::vector< char > & scratchPad
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
	flink::radix_sort( inOutBrickIndices.begin(), inOutBrickIndices.size(), scratchPad );
}

template void svc::Integrator::ExpandBricksHelper< 0 >(Volume const &, Cache &, int, unsigned, flink::vector< unsigned > &, flink::vector< char > &);
template void svc::Integrator::ExpandBricksHelper< 1 >(Volume const &, Cache &, int, unsigned, flink::vector< unsigned > &, flink::vector< char > &);

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
		
	remove_value( inOutBrickIndices, 0u );
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
		
	for( int i = 0; i < volume.Data().size(); i++ )
	{
		unsigned x, y, z;
		flink::unpackInts( volume.Data().keys_first()[ i ], x, y, z );

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

		Voxel vx = volume.Data().values_first()[ i ];
		vx.Update( signedDist, volume.TruncationMargin(), (int) update );
		volume.Data().values_first()[ i ] = vx;
	}
}