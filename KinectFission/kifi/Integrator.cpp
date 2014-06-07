#include <algorithm>
#include <iterator>
#include <utility>
#include <vector>

#include <kifi/util/algorithm.h>
#include <kifi/util/DirectXMathExt.h>
#include <kifi/util/stop_watch.h>
#include <kifi/util/vector2d.h>

#include <kifi/Brick.h>
#include <kifi/Integrator.h>
#include <kifi/Volume.h>
#include <kifi/Voxel.h>



namespace kifi {

// static 
void Integrator::Integrate
( 
	Volume & volume,
	util::vector2d< float > const & frame,
	int footPrint,

	util::float4 const & eye,
	util::float4 const & forward,

	util::float4x4 const & viewProjection,
	util::float4x4 const & viewToWorld
)
{
	assert( footPrint == 2 || footPrint == 4 );

	m_splattedChunks.reserve( frame.size() );
	m_scratchPad.reserve( 4 * frame.size() );

	util::chrono::stop_watch t;
	
	SplatChunks( volume, frame, viewToWorld, footPrint, m_splattedChunks );
	t.take_time( "tsplat" );

	util::radix_sort( m_splattedChunks.begin(), m_splattedChunks.end(), m_scratchPad );
	t.take_time( "tsort" );
	
	m_splattedChunks.resize( 
		std::distance( 
			m_splattedChunks.begin(), 
			std::unique( m_splattedChunks.begin(), m_splattedChunks.end() ) 
		)
	);
	t.take_time( "tdups" );

	ExpandChunks( m_splattedChunks, m_scratchPad );
	t.take_time( "texpand" );

	ChunksToBricks( m_splattedChunks, footPrint, m_scratchPad );
	t.take_time( "tchunk2brick" );

	volume.Data().merge_unique(
		m_splattedChunks.data(), m_splattedChunks.data() + m_splattedChunks.size(), Brick() 
	);
	t.take_time( "tmerge" );

	UpdateVoxels( volume, frame, eye, forward, viewProjection );
	t.take_time( "tupdate" );

	//t.print();
}



// static 
void Integrator::SplatChunks
(
	Volume const & volume,
	util::vector2d< float > const & frame,
	util::float4x4 const & viewToWorld,
	int footPrint,

	std::vector< unsigned > & outChunkIndices
)
{
	outChunkIndices.clear();

	util::mat _viewToWorld = util::load( viewToWorld );

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

		util::float4 pxView
		(
			xNdc * ( halfFrameWidth / fl ) * depth,
			yNdc * ( halfFrameHeight / fl ) * depth,
			-depth,
			1.0f
		);

		util::vec _pxView = util::load( pxView );
		util::vec _pxWorld = _pxView * _viewToWorld;

		util::float4 pxWorld = util::store( _pxWorld );
		util::float4 pxVol = volume.ChunkIndex( pxWorld, footPrint );

		if( pxVol.x < 0.5f ||
			pxVol.y < 0.5f ||
			pxVol.z < 0.5f ||
			
			pxVol.x >= volume.NumChunksInVolume( footPrint ) - 0.5f ||
			pxVol.y >= volume.NumChunksInVolume( footPrint ) - 0.5f ||
			pxVol.z >= volume.NumChunksInVolume( footPrint ) - 0.5f )
			continue;

		outChunkIndices.push_back( util::packInts
		(
			(unsigned) ( pxVol.x - 0.5f ),
			(unsigned) ( pxVol.y - 0.5f ),
			(unsigned) ( pxVol.z - 0.5f )
		));
	}
}

// static 
void Integrator::ExpandChunks
( 
	std::vector< unsigned > & inOutChunkIndices,
	std::vector< char > & tmpScratchPad
)
{
	ExpandChunksHelper( inOutChunkIndices, util::packZ( 1 ), false, tmpScratchPad);
	ExpandChunksHelper( inOutChunkIndices, util::packY( 1 ), false, tmpScratchPad);
	ExpandChunksHelper( inOutChunkIndices, util::packX( 1 ), false, tmpScratchPad);
}

// static 
void Integrator::ChunksToBricks
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
		util::unpackInts( * it, x, y, z );
		* it = util::packInts( 2 * x, 2 * y, 2 * z );
	}

	ExpandChunksHelper( inOutChunkIndices, util::packZ( 1 ), true, tmpScratchPad);
	ExpandChunksHelper( inOutChunkIndices, util::packY( 1 ), true, tmpScratchPad);
	ExpandChunksHelper( inOutChunkIndices, util::packX( 1 ), true, tmpScratchPad);
}

// static
void Integrator::ExpandChunksHelper
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
				newSize -= util::intersection_size
				(
					inOutChunkIndices.cbegin(), inOutChunkIndices.cend(),
					tmp, tmp + inOutChunkIndices.size()
				);

			size_t oldSize = inOutChunkIndices.size();
			inOutChunkIndices.resize( newSize );
	
			util::set_union_backward
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
					std::distance(
						inOutChunkIndices.begin(),
						std::unique( inOutChunkIndices.begin(), inOutChunkIndices.end() )
					)
				);
		}
		break;
	}
}

// static 
void Integrator::UpdateVoxels
(
	Volume & volume,
	util::vector2d< float > const & frame,

	util::float4 const & eye,
	util::float4 const & forward,
	util::float4x4 const & viewProjection
)
{
	util::mat _viewProj = util::load( viewProjection );
	util::vec _ndcToUV = util::set( frame.width() / 2.0f, frame.height() / 2.0f, 0, 0 );

	auto end = volume.Data().keys_cend();
	for
	( 
		auto it = std::make_pair( volume.Data().keys_cbegin(), volume.Data().values_begin() );
		it.first != end;
		++it.first, ++it.second
	)
	{
		unsigned brickX, brickY, brickZ;
		util::unpackInts( * it.first, brickX, brickY, brickZ );

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

			util::float4 centerWorld = volume.VoxelCenter( x, y, z );
			util::vec _centerWorld = util::load( centerWorld );
		
			util::vec _centerNDC = util::homogenize( _centerWorld * _viewProj );
		
			// TODO: Remove SSE4 dependency
			util::vec _centerScreen = _mm_macc_ps( _centerNDC, _ndcToUV, _ndcToUV );
			util::float4 centerScreen = util::store( _centerScreen );

			int u = (int) centerScreen.x;
			int v = (int) centerScreen.y;

			float depth = 0.0f;
			// CUDA: Clamp out of bounds access to 0 to avoid divergence
			if( u >= 0 && u < frame.width() && v >= 0 && v < frame.height() )
				depth = frame( u, frame.height() - v - 1 );

			bool update = ( depth != 0.0f );

			float dist = util::dot( centerWorld - eye, forward );
			float signedDist = depth - dist;
				
			update = update && dist >= 0.8f && signedDist >= -volume.TruncationMargin();

			brick[ j ].Update( signedDist, volume.TruncationMargin(), (int) update );
		}
	}
}

} // namespace