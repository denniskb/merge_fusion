#include <cassert>

#include <kifi/util/math.h>
#include <kifi/util/stop_watch.h>

#include <kifi/Splatter.h>
#include <kifi/Volume.h>



namespace kifi {

// static
void Splatter::Splat( Volume const & volume, std::vector< util::vec3 > & outVertices )
{
	assert( 1 == util::pack( 1, 0, 0 ) );

	outVertices.clear();

	//util::chrono::stop_watch t;

	unsigned deltax = util::pack( 1, 0, 0 );
	unsigned deltay = util::pack( 0, 1, 0 );
	unsigned deltaz = util::pack( 0, 0, 1 );

	auto const keysBegin = volume.Data().keys_cbegin();
	auto const keysEnd   = volume.Data().keys_cend();
	auto const keysLast  = keysEnd - 1;
	auto const values    = volume.Data().values_cbegin();

	auto itSelf  = keysBegin;
	auto itTop   = keysBegin;
	auto itFront = keysBegin;

	for( ; itSelf != keysEnd; ++itSelf )
	{
		Voxel self  = values[ itSelf - keysBegin ];

		if( 0.0f == self.Weight() )
			continue;

		while( * itTop < * itSelf + deltay && itTop < keysLast )
			++itTop;

		while( * itFront < * itSelf + deltaz && itFront < keysLast )
			++itFront;

		Voxel right = 
			( itSelf < keysLast && * (itSelf + 1) == * itSelf + deltax ) ? 
			values[ itSelf - keysBegin + 1 ] : Voxel();

		Voxel top = 
			( * itTop == * itSelf + deltay ) ? 
			values[ itTop - keysBegin ] : Voxel();

		Voxel front = 
			( * itFront == * itSelf + deltaz ) ? 
			values[ itFront - keysBegin ] : Voxel();

		unsigned x, y, z;
		util::unpack( * itSelf, x, y, z );

		util::vec3 vert000 = volume.VoxelCenter( x, y, z ); 

		float dself, dright, dtop, dfront;
		dself  = self. Distance();
		dright = right.Distance();
		dtop   = top.  Distance();
		dfront = front.Distance();

		if( right.Weight() > 0.0f && dself * dright < 0.0f )
		{
			util::vec3 vert = vert000;
			vert.x += util::lerp( 0.0f, volume.VoxelLength(), abs(dself) / (abs(dself) + abs(dright)) );

			outVertices.push_back( vert );
		}
				
		if( top.Weight() > 0.0f && dself * dtop < 0.0f )
		{
			util::vec3 vert = vert000;
			vert.y += util::lerp( 0.0f, volume.VoxelLength(), abs(dself) / (abs(dself) + abs(dtop)) );

			outVertices.push_back( vert );
		}
				
		if( front.Weight() > 0.0f && dself * dfront < 0.0f )
		{
			util::vec3 vert = vert000;
			vert.z += util::lerp( 0.0f, volume.VoxelLength(), abs(dself) / (abs(dself) + abs(dfront)) );

			outVertices.push_back( vert );
		}
	}

	//t.take_time( "tsplat" );
	//t.print_times();
}

} // namespace