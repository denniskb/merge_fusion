#include "Integrator.h"

#include <utility>
#include <vector>

#include "algorithm.h"
#include "Brick.h"
#include "dxmath.h"
#include "vector2d.h"
#include "Volume.h"
#include "Voxel.h"

#include "timer.h"



// static 
void svc::Integrator::Integrate
( 
	Volume & volume,
	vector2d< float > const & frame,
	int footPrint,

	float4 const & eye,
	float4 const & forward,

	float4x4 const & viewProjection,
	float4x4 const & viewToWorld
)
{
	assert( footPrint == 2 || footPrint == 4 );

	m_splattedChunks.reserve( frame.size() );
	m_scratchPad.reserve( 4 * frame.size() );

	timer t;
	
	SplatChunks( volume, frame, viewToWorld, footPrint, m_splattedChunks );
	t.record_time( "tsplat" );

	radix_sort( m_splattedChunks.begin(), m_splattedChunks.end(), m_scratchPad );
	t.record_time( "tsort" );
	
	m_splattedChunks.resize( svc::unique( m_splattedChunks.begin(), m_splattedChunks.end() ) );
	t.record_time( "tdups" );

	ExpandChunks( m_splattedChunks, m_scratchPad );
	t.record_time( "texpand" );

	ChunksToBricks( m_splattedChunks, footPrint, m_scratchPad );
	t.record_time( "tchunk2brick" );

	volume.Data().merge_unique(
		m_splattedChunks.data(), m_splattedChunks.data() + m_splattedChunks.size(), Brick() 
	);
	t.record_time( "tmerge" );

	UpdateVoxels( volume, frame, eye, forward, viewProjection );
	t.record_time( "tupdate" );

	//t.print();
}



// static 
void svc::Integrator::SplatChunks
(
	Volume const & volume,
	vector2d< float > const & frame,
	float4x4 const & viewToWorld,
	int footPrint,

	std::vector< unsigned > & outChunkIndices
)
{
	outChunkIndices.clear();

	mat _viewToWorld = load( viewToWorld );

	float const halfFrameWidth = (float) ( frame.width() / 2 );
	float const halfFrameHeight = (float) ( frame.height() / 2 );

	float const ppX = halfFrameWidth - 0.5f;
	float const ppY = halfFrameHeight - 0.5f;

	// TODO: Encapsulate in CameraParams struct!
	float const fl = 585.0f;

	for( int i = 0; i < frame.size(); ++i )
	{
		float depth = frame[ i ];
		if( 0.0f == depth )
			continue;

		int y = (int) ( i / frame.width() );
		int x = (int) ( i % frame.width() );

		float xNdc = ( x - ppX ) / halfFrameWidth;
		float yNdc = ( ppY - y ) / halfFrameHeight;

		float4 pxView
		(
			xNdc * ( halfFrameWidth / fl ) * depth,
			yNdc * ( halfFrameHeight / fl ) * depth,
			-depth,
			1.0f
		);

		vec _pxView = load( pxView );
		vec _pxWorld = _pxView * _viewToWorld;

		float4 pxWorld = store( _pxWorld );
		float4 pxVol = volume.ChunkIndex( pxWorld, footPrint );

		if( pxVol.x < 0.5f ||
			pxVol.y < 0.5f ||
			pxVol.z < 0.5f ||
			
			pxVol.x >= volume.NumChunksInVolume( footPrint ) - 0.5f ||
			pxVol.y >= volume.NumChunksInVolume( footPrint ) - 0.5f ||
			pxVol.z >= volume.NumChunksInVolume( footPrint ) - 0.5f )
			continue;

		outChunkIndices.push_back( packInts
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
	std::vector< unsigned > & inOutChunkIndices,
	std::vector< char > & tmpScratchPad
)
{
	ExpandChunksHelper( inOutChunkIndices, packZ( 1 ), false, tmpScratchPad);
	ExpandChunksHelper( inOutChunkIndices, packY( 1 ), false, tmpScratchPad);
	ExpandChunksHelper( inOutChunkIndices, packX( 1 ), false, tmpScratchPad);
}

// static 
void svc::Integrator::ChunksToBricks
(
	std::vector< unsigned > & inOutChunkIndices,
	int footPrint,

	std::vector< char > & tmpScratchPad
)
{
	if( footPrint != 4 )
		return;

	for( auto it = inOutChunkIndices.begin(); it != inOutChunkIndices.end(); ++it )
	{
		unsigned x, y, z;
		unpackInts( * it, x, y, z );
		* it = packInts( 2 * x, 2 * y, 2 * z );
	}

	ExpandChunksHelper( inOutChunkIndices, packZ( 1 ), true, tmpScratchPad);
	ExpandChunksHelper( inOutChunkIndices, packY( 1 ), true, tmpScratchPad);
	ExpandChunksHelper( inOutChunkIndices, packX( 1 ), true, tmpScratchPad);
}

// static
void svc::Integrator::ExpandChunksHelper
(
	std::vector< unsigned > & inOutChunkIndices,
	unsigned delta,
	bool disjunct,

	std::vector< char > & tmpScratchPad
)
{
	switch( delta )
	{
	default:
		{
			tmpScratchPad.resize( inOutChunkIndices.size() * sizeof( unsigned ) );
			unsigned * tmp = reinterpret_cast< unsigned * >( tmpScratchPad.data() );
			
			for( auto it = std::make_pair( inOutChunkIndices.cbegin(), tmp );
				 it.first != inOutChunkIndices.cend();
				 ++it.first, ++it.second )
				* it.second = * it.first + delta;
	
			size_t newSize = 2 * inOutChunkIndices.size();
			if( ! disjunct )
				newSize -= intersection_size
				(
					inOutChunkIndices.cbegin(), inOutChunkIndices.cend(),
					tmp, tmp + inOutChunkIndices.size()
				);

			size_t oldSize = inOutChunkIndices.size();
			inOutChunkIndices.resize( newSize );
	
			set_union_backward
			(
				inOutChunkIndices.cbegin(), inOutChunkIndices.cbegin() + oldSize,
				tmp, tmp + oldSize,
		
				inOutChunkIndices.end()
			);
		}
		break;

	case 1:
		{
			size_t oldSize = inOutChunkIndices.size();
			inOutChunkIndices.resize( 2 * oldSize );
	
			for( size_t i = 0; i < oldSize; ++i )
			{
				size_t ii = oldSize - i - 1;
				
				unsigned tmp = inOutChunkIndices[ ii ];
				ii *= 2;
				inOutChunkIndices[ ii ] = tmp;
				inOutChunkIndices[ ii + 1 ] = tmp + 1;
			}

			if( ! disjunct )
				inOutChunkIndices.resize( 
					svc::unique( inOutChunkIndices.begin(), inOutChunkIndices.end() )
				);
		}
		break;
	}
}

// static 
void svc::Integrator::UpdateVoxels
(
	Volume & volume,
	vector2d< float > const & frame,

	float4 const & eye,
	float4 const & forward,
	float4x4 const & viewProjection
)
{
	mat _viewProj = load( viewProjection );
	vec _ndcToUV = set( frame.width() / 2.0f, frame.height() / 2.0f, 0, 0 );

	auto end = volume.Data().keys_cend();
	for
	( 
		auto it = std::make_pair( volume.Data().keys_cbegin(), volume.Data().values_begin() );
		it.first != end;
		++it.first, ++it.second
	)
	{
		unsigned brickX, brickY, brickZ;
		unpackInts( * it.first, brickX, brickY, brickZ );

		brickX *= 2;
		brickY *= 2;
		brickZ *= 2;

		Brick & brick = * it.second;

		for( int j = 0; j < brick.size(); ++j )
		{
			unsigned x, y, z;
			Brick::Index1Dto3D( j, x, y, z );

			x += brickX;
			y += brickY;
			z += brickZ;

			float4 centerWorld = volume.VoxelCenter( x, y, z );
			vec _centerWorld = load( centerWorld );
		
			vec _centerNDC = homogenize( _centerWorld * _viewProj );
		
			vec _centerScreen = _mm_macc_ps( _centerNDC, _ndcToUV, _ndcToUV );
			float4 centerScreen = store( _centerScreen );

			int u = (int) centerScreen.x;
			int v = (int) centerScreen.y;

			float depth = 0.0f;
			// CUDA: Clamp out of bounds access to 0 to avoid divergence
			if( u >= 0 && u < frame.width() && v >= 0 && v < frame.height() )
				depth = frame( u, frame.height() - v - 1 );

			bool update = ( depth != 0.0f );

			float dist = dot( centerWorld - eye, forward );
			float signedDist = depth - dist;
				
			update = update && dist >= 0.8f && signedDist >= -volume.TruncationMargin();

			brick[ j ].Update( signedDist, volume.TruncationMargin(), (int) update );
		}
	}
}