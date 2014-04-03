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
	DepthFrame const & frame,

	flink::float4 const & eye,
	flink::float4 const & forward,

	flink::float4x4 const & viewProjection,
	flink::float4x4 const & viewToWorld
)
{
	flink::timer t;
	
	SplatBricks( volume, frame, viewToWorld, m_splattedVoxels );
	t.record_time( "tsplat" );

	flink::radix_sort( m_splattedVoxels.begin(), m_splattedVoxels.size(), m_scratchPad );
	t.record_time( "tsort" );
	
	remove_dups( m_splattedVoxels );
	t.record_time( "tdups" );

	ExpandBricks( m_splattedVoxels, m_scratchPad );
	t.record_time( "texpand" );
	
	BricksToVoxels( volume, m_splattedVoxels );
	t.record_time( "tbricks" );
	
	flink::radix_sort( m_splattedVoxels.begin(), m_splattedVoxels.size(), m_scratchPad );
	t.record_time( "tsort2" );

	volume.Data().merge_unique( m_splattedVoxels.cbegin(), m_splattedVoxels.cend(), 0 );
	t.record_time( "tmerge" );

	UpdateVoxels( volume, frame, eye, forward, viewProjection );
	t.record_time( "tupdate" );

	t.print();
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
	flink::vector< unsigned > & inOutBrickIndices,
	flink::vector< char > & tmpScratchPad
)
{
	ExpandBricksHelper( inOutBrickIndices, flink::packX( 1 ), tmpScratchPad );
	ExpandBricksHelper( inOutBrickIndices, flink::packY( 1 ), tmpScratchPad );
	ExpandBricksHelper( inOutBrickIndices, flink::packZ( 1 ), tmpScratchPad );
}

// static
void svc::Integrator::ExpandBricksHelper
(
	flink::vector< unsigned > & inOutBrickIndices,
	unsigned delta,

	flink::vector< char > & tmpScratchPad
)
{
	int size = inOutBrickIndices.size();
	
	tmpScratchPad.resize( size * sizeof( unsigned ) );
	unsigned * tmp = reinterpret_cast< unsigned * >( tmpScratchPad.begin() );
	
	for( int i = 0; i < size; i++ )
		tmp[ i ] = inOutBrickIndices[ i ] + delta;
	
	inOutBrickIndices.resize( size * 2 - flink::intersection_size
	(
		inOutBrickIndices.cbegin(), inOutBrickIndices.cbegin() + size,
		tmp, tmp + size
	));
	
	flink::merge_unique_backward
	(
		inOutBrickIndices.cbegin(), inOutBrickIndices.cbegin() + size,
		tmp, tmp + size,
		
		inOutBrickIndices.end()
	);
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