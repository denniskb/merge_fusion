#include "Integrator.h"

#include <flink/algorithm.h>
#include <flink/util.h>
#include <flink/vector.h>

#include "Brick.h"
#include "DepthFrame.h"
#include "Volume.h"
#include "Voxel.h"

#include <flink/timer.h>



// static 
void svc::Integrator::Integrate
( 
	Volume & volume,
	DepthFrame const & frame,
	int footPrint,

	flink::float4 const & eye,
	flink::float4 const & forward,

	flink::float4x4 const & viewProjection,
	flink::float4x4 const & viewToWorld
)
{
	assert( footPrint == 2 || footPrint == 4 );

	flink::timer t;
	
	SplatChunks( volume, frame, viewToWorld, footPrint, m_splattedChunks );
	t.record_time( "tsplat" );

	flink::radix_sort( m_splattedChunks.begin(), m_splattedChunks.size(), m_scratchPad );
	t.record_time( "tsort" );
	
	remove_dups( m_splattedChunks );
	t.record_time( "tdups" );

	ExpandChunks( m_splattedChunks, m_scratchPad );
	t.record_time( "texpand" );

	ChunksToBricks( m_splattedChunks, footPrint, m_scratchPad );
	t.record_time( "tchunk2brick" );

	volume.Data().merge_unique( m_splattedChunks.cbegin(), m_splattedChunks.cend(), Brick() );
	t.record_time( "tmerge" );

	UpdateVoxels( volume, frame, eye, forward, viewProjection );
	t.record_time( "tupdate" );

	t.print();
}



// static 
void svc::Integrator::SplatChunks
(
	Volume const & volume,
	DepthFrame const & frame,
	flink::float4x4 const & viewToWorld,
	int footPrint,

	flink::vector< unsigned > & outChunkIndices
)
{
	outChunkIndices.clear();

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
		flink::float4 pxVol = volume.ChunkIndex( pxWorld, footPrint );

		if( pxVol < flink::make_float4( 0.5f ) ||
			pxVol >= flink::make_float4( volume.NumChunksInVolume( footPrint ) - 0.5f ) )
			continue;

		outChunkIndices.push_back( flink::packInts
		(
			(unsigned) ( pxVol.x - 0.5f ),
			(unsigned) ( pxVol.y - 0.5f ),
			(unsigned) ( pxVol.z - 0.5f )
		));
	}
}

// static 
void svc::Integrator::ExpandChunks
( 
	flink::vector< unsigned > & inOutChunkIndices,
	flink::vector< char > & tmpScratchPad
)
{
	ExpandChunksHelper( inOutChunkIndices, flink::packZ( 1 ), false, tmpScratchPad);
	ExpandChunksHelper( inOutChunkIndices, flink::packY( 1 ), false, tmpScratchPad);
	ExpandChunksHelper( inOutChunkIndices, flink::packX( 1 ), false, tmpScratchPad);
}

// static 
void svc::Integrator::ChunksToBricks
(
	flink::vector< unsigned > & inOutChunkIndices,
	int footPrint,

	flink::vector< char > & tmpScratchPad
)
{
	if( footPrint != 4 )
		return;

	for( int i = 0; i < inOutChunkIndices.size(); i++ )
	{
		unsigned x, y, z;
		flink::unpackInts( inOutChunkIndices[ i ], x, y, z );
		inOutChunkIndices[ i ] = flink::packInts( 2 * x, 2 * y, 2 * z );
	}

	ExpandChunksHelper( inOutChunkIndices, flink::packZ( 1 ), true, tmpScratchPad);
	ExpandChunksHelper( inOutChunkIndices, flink::packY( 1 ), true, tmpScratchPad);
	ExpandChunksHelper( inOutChunkIndices, flink::packX( 1 ), true, tmpScratchPad);
}

// static
void svc::Integrator::ExpandChunksHelper
(
	flink::vector< unsigned > & inOutChunkIndices,
	unsigned delta,
	bool disjunct,

	flink::vector< char > & tmpScratchPad
)
{
	switch( delta )
	{
	default:
		{
			int oldSize = inOutChunkIndices.size();

			tmpScratchPad.resize( oldSize * sizeof( unsigned ) );
			unsigned * tmp = reinterpret_cast< unsigned * >( tmpScratchPad.begin() );
	
			for( int i = 0; i < oldSize; i++ )
				tmp[ i ] = inOutChunkIndices[ i ] + delta;
	
			int newSize;
			if( disjunct )
				newSize = 2 * oldSize;
			else
				newSize = 2 * oldSize - flink::intersection_size(
					inOutChunkIndices.cbegin(), inOutChunkIndices.cbegin() + oldSize,
					tmp, tmp + oldSize
				);

			inOutChunkIndices.resize( newSize );
	
			flink::merge_unique_backward
			(
				inOutChunkIndices.cbegin(), inOutChunkIndices.cbegin() + oldSize,
				tmp, tmp + oldSize,
		
				inOutChunkIndices.end()
			);
		}
		break;

	case 1:
		{
			int oldSize = inOutChunkIndices.size();
			inOutChunkIndices.resize( 2 * oldSize );
	
			for( int i = oldSize - 1; i >= 0; i-- )
			{
				unsigned tmp = inOutChunkIndices[ i ];
				inOutChunkIndices[ 2 * i ] = tmp;
				inOutChunkIndices[ 2 * i + 1 ] = tmp + 1;
			}

			if( ! disjunct )
				flink::remove_dups( inOutChunkIndices );
		}
		break;
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
		unsigned brickX, brickY, brickZ;
		flink::unpackInts( volume.Data().keys_first()[ i ], brickX, brickY, brickZ );

		brickX *= 2;
		brickY *= 2;
		brickZ *= 2;

		Brick & brick = volume.Data().values_first()[ i ];

		for( int j = 0; j < brick.size(); j++ )
		{
			unsigned x, y, z;
			Brick::Index1Dto3D( j, x, y, z );

			x += brickX;
			y += brickY;
			z += brickZ;

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
				
			update = update && dist >= 0.8f && signedDist >= -volume.TruncationMargin();

			brick[ j ].Update( signedDist, volume.TruncationMargin(), (int) update );
		}
	}
}