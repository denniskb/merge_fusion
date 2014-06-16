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



// HACK
#include <kifi/util/stop_watch.h>



namespace kifi {

// static 
void Integrator::Integrate
( 
	Volume & volume,
	util::vector2d< float > const & frame,

	util::vec3 eye,
	util::vec3 forward,

	util::matrix const & viewProjection,
	util::matrix4x3 const & viewToWorld
)
{
	m_tmpPointCloud.resize ( frame.size() );
	m_tmpScratchPad.reserve( frame.size() );

	util::chrono::stop_watch sw;
	size_t nSplats = DepthMap2PointCloud( volume, frame, viewToWorld, m_tmpPointCloud );
	sw.take_time( "tsplat" );

	util::radix_sort( m_tmpPointCloud.data(), m_tmpPointCloud.data() + nSplats, m_tmpScratchPad.data() );
	sw.take_time( "tsort" );

	m_tmpPointCloud.resize(
		std::distance( 
			m_tmpPointCloud.begin(), 
			std::unique( m_tmpPointCloud.begin(), m_tmpPointCloud.begin() + nSplats ) 
		)
	);
	sw.take_time( "unique" );

	ExpandChunks( m_tmpPointCloud, m_tmpScratchPad );
	sw.take_time( "expand" );
	
	volume.Data().merge_unique(
		m_tmpPointCloud.data(), m_tmpPointCloud.data() + m_tmpPointCloud.size(), Voxel()
	);
	sw.take_time( "merge" );

	UpdateVoxels( volume, frame, eye, forward, viewProjection );
	sw.take_time( "tupdate" );

	sw.print_times();
}



// static 
size_t Integrator::DepthMap2PointCloud
(
	Volume const & volume,
	util::vector2d< float > const & frame,
	util::matrix4x3 const & viewToWorld,

	std::vector< unsigned > & outPointCloud
)
{
	// TODO: Encapsulate in CameraParams struct!
	float const flInv = 1.0f / 585.0f;
	float const nppxOverFl = -((frame.width()  / 2) - 0.5f) * flInv;
	float const  ppyOverFl =  ((frame.height() / 2) - 0.5f) * flInv;
	int const maxIndex = volume.Resolution() - 1;

	size_t nSplats = 0;

	for( size_t v = 0; v < frame.height(); v++ )
		for( size_t u = 0; u < frame.width(); u++ )
		{
			float depth = frame( u, v );
			if( 0.0f == depth )
				continue;

			util::vec3 pxView
			(
				(   (float) u  * flInv + nppxOverFl ) * depth,
				( -((float) v) * flInv +  ppyOverFl ) * depth,
				-depth
			);

			util::vec3 pxWorld = util::transform_point( pxView, viewToWorld );
			util::vec3 pxVolume = volume.VoxelIndex( pxWorld ) - 0.5f;

			int x, y, z;
			x = (int) pxVolume.x;
			y = (int) pxVolume.y;
			z = (int) pxVolume.z;
		
			if( x < 0 ||
				y < 0 ||
				z < 0 ||
			
				x >= maxIndex ||
				y >= maxIndex ||
				z >= maxIndex )
				continue;
			
			outPointCloud[ nSplats++ ] = util::pack( x, y, z );
		}

	return nSplats;
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

	util::vec3 eye,
	util::vec3 forward,
	util::matrix const & viewProjection
)
{
	float const ndcToUVx = frame.width() * 0.5f;
	float const ndcToUVy = frame.height() * 0.5f;

	auto end = volume.Data().keys_cend();
	for
	( 
		auto it = std::make_pair( volume.Data().keys_cbegin(), volume.Data().values_begin() );
		it.first != end;
		++it.first, ++it.second
	)
	{
		uint32_t x, y, z;
		util::unpack( * it.first, x, y, z );

		util::vec3 centerWorld = volume.VoxelCenter( x, y, z );
		float dist = util::dot( centerWorld - eye, forward );

		util::vec3 centerNDC = util::project( centerWorld, viewProjection );
		int u = (int) ( centerNDC.x * ndcToUVx + ndcToUVx );
		int v = (int) ( centerNDC.y * ndcToUVy + ndcToUVy );
		
		if( u < 0 || u >= frame.width() || v < 0 || v >= frame.height() )
			continue;

		float depth = frame( u, frame.height() - v - 1 );
		float signedDist = depth - dist;
				
		int update = ( signedDist >= -volume.TruncationMargin() && 0.0f != depth && dist >= 0.8f );

		it.second->Update( std::min(signedDist, volume.TruncationMargin()), (float)update );
	}
}

} // namespace