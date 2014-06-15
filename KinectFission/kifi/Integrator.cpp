#include <algorithm>
#include <iterator>
#include <utility>
#include <vector>

#include <kifi/util/algorithm.h>
#include <kifi/util/math.h>
#include <kifi/util/iterator.h>
#include <kifi/util/vector2d.h>

#include <kifi/Integrator.h>
#include <kifi/Volume.h>
#include <kifi/Voxel.h>



// HACK during dev
#pragma warning( disable : 4100 )



namespace {

struct delta
{
	delta( unsigned n ) :
		m_n( n )
	{}

	unsigned operator()( unsigned x ) const
	{
		return x + m_n;
	}

private:
	unsigned m_n;
};

}



namespace kifi {

// static 
void Integrator::Integrate
( 
	Volume & volume,
	util::vector2d< float > const & frame,

	util::float4 const & eye,
	util::float4 const & forward,

	util::float4x4 const & viewProjection,
	util::float4x4 const & viewToWorld
)
{
	m_tmpPointCloud.reserve( frame.size() );
	m_tmpScratchPad.reserve( frame.size() );

	DepthMap2PointCloud( volume, frame, viewToWorld, m_tmpPointCloud );

	util::radix_sort( m_tmpPointCloud.data(), m_tmpPointCloud.data() + m_tmpPointCloud.size(), m_tmpScratchPad.data() );
	
	m_tmpPointCloud.resize( 
		std::distance( 
			m_tmpPointCloud.begin(), 
			std::unique( m_tmpPointCloud.begin(), m_tmpPointCloud.end() ) 
		)
	);

	ExpandChunks( m_tmpPointCloud, m_tmpScratchPad );
	
	volume.Data().merge_unique(
		m_tmpPointCloud.data(), m_tmpPointCloud.data() + m_tmpPointCloud.size(), 0 
	);

	UpdateVoxels( volume, frame, eye, forward, viewProjection );
}



// static 
void Integrator::DepthMap2PointCloud
(
	Volume const & volume,
	util::vector2d< float > const & frame,
	util::float4x4 const & viewToWorld,

	std::vector< unsigned > & outPointCloud
)
{
	outPointCloud.clear();

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

		int v = i / (int)frame.width();
		int u = i - v * (int)frame.width();

		float uNdc = ( u - ppX ) / halfFrameWidth;
		float vNdc = ( ppY - v ) / halfFrameHeight;

		util::float4 pxView
		(
			uNdc * ( halfFrameWidth / fl ) * depth,
			vNdc * ( halfFrameHeight / fl ) * depth,
			-depth,
			1.0f
		);

		util::float4 pxWorld = pxView * viewToWorld;
		util::float4 pxVol = volume.VoxelIndex( pxWorld );

		int x, y, z;
		x = (int) ( pxVol.x - 0.5f );
		y = (int) ( pxVol.y - 0.5f );
		z = (int) ( pxVol.z - 0.5f );
		
		int maxIndex = volume.Resolution() - 1;
		if( x < 0 ||
			y < 0 ||
			z < 0 ||
			
			x >= maxIndex ||
			y >= maxIndex ||
			z >= maxIndex )
			continue;

		outPointCloud.push_back( util::pack( x, y, z ) );
	}
}

// static 
void Integrator::ExpandChunks
( 
	std::vector< unsigned > & inOutChunkIndices,
	std::vector< unsigned > & tmpScratchPad
)
{
	ExpandChunksHelper( inOutChunkIndices, util::pack( 0, 0, 1 ), tmpScratchPad);
	ExpandChunksHelper( inOutChunkIndices, util::pack( 0, 1, 0 ), tmpScratchPad);
	ExpandChunksHelper( inOutChunkIndices, util::pack( 1, 0, 0 ), tmpScratchPad);
}

// static
void Integrator::ExpandChunksHelper
(
	std::vector< unsigned > & inOutChunkIndices,
	unsigned delta,

	std::vector< unsigned > & tmpScratchPad
)
{
	if( 0 == inOutChunkIndices.size() )
		return;

	switch( delta )
	{
	default:
		{
			tmpScratchPad.resize( 2 * inOutChunkIndices.size() );

			auto tmpNewEnd = std::set_union
			(
				inOutChunkIndices.cbegin(), inOutChunkIndices.cend(),
				util::make_map_iterator( inOutChunkIndices.cbegin(), ::delta( delta ) ), util::make_map_iterator( inOutChunkIndices.cend(), ::delta( delta ) ),
				tmpScratchPad.data()
			);

			tmpScratchPad.resize( std::distance( tmpScratchPad.data(), tmpNewEnd ) );
			std::swap( tmpScratchPad, inOutChunkIndices );
		}
		break;

	case 1:
		{
			tmpScratchPad.clear();
			unsigned prev = inOutChunkIndices[ 0 ];

			for( std::size_t i = 0; i < inOutChunkIndices.size(); ++i )
			{
				unsigned x = inOutChunkIndices[ i ];

				if( x != prev )
					tmpScratchPad.push_back( prev );
				
				tmpScratchPad.push_back( x );

				prev = x + 1;
			}

			std::swap( tmpScratchPad, inOutChunkIndices );
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
	util::float4 ndcToUV( frame.width() * 0.5f, frame.height() * 0.5f, 0.0f, 0.0f );

	auto end = volume.Data().keys_cend();
	for
	( 
		auto it = std::make_pair( volume.Data().keys_cbegin(), volume.Data().values_begin() );
		it.first != end;
		++it.first, ++it.second
	)
	{
		unsigned x, y, z;
		util::unpack( * it.first, x, y, z );

		util::float4 centerWorld = volume.VoxelCenter( x, y, z );
		util::float4 centerNDC = util::homogenize( centerWorld * viewProjection );
		util::float4 centerScreen = centerNDC * ndcToUV + ndcToUV;

		int u = (int) centerScreen.x;
		int v = (int) centerScreen.y;

		float depth = 0.0f;
		// CUDA: Clamp out of bounds access to 0 to avoid divergence
		if( u >= 0 && u < frame.width() && v >= 0 && v < frame.height() )
			depth = frame( u, frame.height() - v - 1 );

		float dist = util::dot( centerWorld - eye, forward );
		float signedDist = depth - dist;
				
		int update = ( signedDist >= -volume.TruncationMargin() ) && ( depth != 0.0f ) && ( dist >= 0.8f );

		it.second->Update( signedDist, volume.TruncationMargin(), update );
	}
}

} // namespace