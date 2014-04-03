#include "Mesher.h"

#include <flink/algorithm.h>
#include <flink/util.h>

#include "Cache.h"
#include "Volume.h"
#include "Voxel.h"

#include <flink/timer.h>



template< int BrickRes >
void svc::Mesher::Triangulate
(
	Volume< BrickRes > const & volume,
	Cache & cache,

	flink::vector< flink::float4 > & outVertices,
	flink::vector< unsigned > & outIndices
)
{
	flink::timer t;
	
	Generate( volume, cache, outVertices, m_vertexIDs, outIndices );	
	t.record_time( "tgen" );

	flink::radix_sort( m_vertexIDs.begin(), outVertices.begin(), m_vertexIDs.size(), m_scratchPad );
	t.record_time( "tsort" );
	
	VertexIDsToIndices( m_vertexIDs, outIndices, m_indexIDs, m_scratchPad );
	t.record_time( "tidx" );

	t.print();
}



// static 
void svc::Mesher::Mesh2Obj
(
	flink::vector< flink::float4 > const & vertices,
	flink::vector< unsigned > const & indices,

	char const * outObjFileName
)
{
	FILE * file;
	fopen_s( & file, outObjFileName, "w" );

	for( int i = 0; i < vertices.size(); i++ )
	{
		auto v = vertices[ i ];
		fprintf_s( file, "v %f %f %f\n", v.x, v.y, v.z );
	}

	for( int i = 0; i < indices.size(); i += 3 )
	{
		fprintf_s
		(
			file, 
			"f %d %d %d\n", 
			indices[ i ] + 1, indices[ i + 1 ] + 1, indices[ i + 2 ] + 1
		);
	}

	fclose( file );
}



// static
int const * svc::Mesher::TriOffsets()
{
	static int const triCounts[] = {
		0,   0,   1,   2,   4,   5,   7,   9,  12,  13,  15,  17,  20,  22,  25,  28,
	   30,  31,  33,  35,  38,  40,  43,  46,  50,  52,  55,  58,  62,  65,  69,  73,
	   76,  77,  79,  81,  84,  86,  89,  92,  96,  98, 101, 104, 108, 111, 115, 119,
	  122, 124, 127, 130, 132, 135, 139, 143, 146, 149, 153, 157, 160, 164, 169, 174,
	  176, 177, 179, 181, 184, 186, 189, 192, 196, 198, 201, 204, 208, 211, 215, 219,
	  222, 224, 227, 230, 234, 237, 241, 245, 250, 253, 257, 261, 266, 270, 275, 280,
	  284, 286, 289, 292, 296, 299, 303, 305, 308, 311, 315, 319, 324, 328, 333, 336,
	  338, 341, 345, 349, 352, 356, 361, 364, 366, 370, 375, 380, 384, 389, 391, 395,
	  396, 397, 399, 401, 404, 406, 409, 412, 416, 418, 421, 424, 428, 431, 435, 439,
	  442, 444, 447, 450, 454, 457, 461, 465, 470, 473, 475, 479, 482, 486, 489, 494,
	  496, 498, 501, 504, 508, 511, 515, 519, 524, 527, 531, 535, 540, 544, 549, 554,
	  558, 561, 565, 569, 572, 576, 581, 586, 590, 594, 597, 602, 604, 609, 613, 615,
	  616, 618, 621, 624, 628, 631, 635, 639, 644, 647, 651, 655, 660, 662, 665, 668,
	  670, 673, 677, 681, 686, 690, 695, 700, 702, 706, 709, 714, 718, 721, 723, 727,
	  728, 731, 735, 739, 744, 748, 753, 756, 760, 764, 769, 774, 776, 779, 783, 785,
	  786, 788, 791, 794, 796, 799, 803, 805, 806, 809, 811, 815, 816, 818, 819, 820
	};

	return triCounts;
}

// static
flink::uint4 const * svc::Mesher::TriTable()
{
	static unsigned const triTable[] = {
		 0,  8,  3, 0,  0,  1,  9, 0,  1,  8,  3, 0,  9,  8,  1, 0,
		 1,  2, 10, 0,  0,  8,  3, 0,  1,  2, 10, 0,  9,  2, 10, 0,
		 0,  2,  9, 0,  2,  8,  3, 0,  2, 10,  8, 0, 10,  9,  8, 0,
		 3, 11,  2, 0,  0, 11,  2, 0,  8, 11,  0, 0,  1,  9,  0, 0,
		 2,  3, 11, 0,  1, 11,  2, 0,  1,  9, 11, 0,  9,  8, 11, 0,
		 3, 10,  1, 0, 11, 10,  3, 0,  0, 10,  1, 0,  0,  8, 10, 0,
		 8, 11, 10, 0,  3,  9,  0, 0,  3, 11,  9, 0, 11, 10,  9, 0,
		 9,  8, 10, 0, 10,  8, 11, 0,  4,  7,  8, 0,  4,  3,  0, 0,
		 7,  3,  4, 0,  0,  1,  9, 0,  8,  4,  7, 0,  4,  1,  9, 0,
		 4,  7,  1, 0,  7,  3,  1, 0,  1,  2, 10, 0,  8,  4,  7, 0,
		 3,  4,  7, 0,  3,  0,  4, 0,  1,  2, 10, 0,  9,  2, 10, 0,
		 9,  0,  2, 0,  8,  4,  7, 0,  2, 10,  9, 0,  2,  9,  7, 0,
		 2,  7,  3, 0,  7,  9,  4, 0,  8,  4,  7, 0,  3, 11,  2, 0,
		11,  4,  7, 0, 11,  2,  4, 0,  2,  0,  4, 0,  9,  0,  1, 0,
		 8,  4,  7, 0,  2,  3, 11, 0,  4,  7, 11, 0,  9,  4, 11, 0,
		 9, 11,  2, 0,  9,  2,  1, 0,  3, 10,  1, 0,  3, 11, 10, 0,
		 7,  8,  4, 0,  1, 11, 10, 0,  1,  4, 11, 0,  1,  0,  4, 0,
		 7, 11,  4, 0,  4,  7,  8, 0,  9,  0, 11, 0,  9, 11, 10, 0,
		11,  0,  3, 0,  4,  7, 11, 0,  4, 11,  9, 0,  9, 11, 10, 0,
		 9,  5,  4, 0,  9,  5,  4, 0,  0,  8,  3, 0,  0,  5,  4, 0,
		 1,  5,  0, 0,  8,  5,  4, 0,  8,  3,  5, 0,  3,  1,  5, 0,
		 1,  2, 10, 0,  9,  5,  4, 0,  3,  0,  8, 0,  1,  2, 10, 0,
		 4,  9,  5, 0,  5,  2, 10, 0,  5,  4,  2, 0,  4,  0,  2, 0,
		 2, 10,  5, 0,  3,  2,  5, 0,  3,  5,  4, 0,  3,  4,  8, 0,
		 9,  5,  4, 0,  2,  3, 11, 0,  0, 11,  2, 0,  0,  8, 11, 0,
		 4,  9,  5, 0,  0,  5,  4, 0,  0,  1,  5, 0,  2,  3, 11, 0,
		 2,  1,  5, 0,  2,  5,  8, 0,  2,  8, 11, 0,  4,  8,  5, 0,
		10,  3, 11, 0, 10,  1,  3, 0,  9,  5,  4, 0,  4,  9,  5, 0,
		 0,  8,  1, 0,  8, 10,  1, 0,  8, 11, 10, 0,  5,  4,  0, 0,
		 5,  0, 11, 0,  5, 11, 10, 0, 11,  0,  3, 0,  5,  4,  8, 0,
		 5,  8, 10, 0, 10,  8, 11, 0,  9,  7,  8, 0,  5,  7,  9, 0,
		 9,  3,  0, 0,  9,  5,  3, 0,  5,  7,  3, 0,  0,  7,  8, 0,
		 0,  1,  7, 0,  1,  5,  7, 0,  1,  5,  3, 0,  3,  5,  7, 0,
		 9,  7,  8, 0,  9,  5,  7, 0, 10,  1,  2, 0, 10,  1,  2, 0,
		 9,  5,  0, 0,  5,  3,  0, 0,  5,  7,  3, 0,  8,  0,  2, 0,
		 8,  2,  5, 0,  8,  5,  7, 0, 10,  5,  2, 0,  2, 10,  5, 0,
		 2,  5,  3, 0,  3,  5,  7, 0,  7,  9,  5, 0,  7,  8,  9, 0,
		 3, 11,  2, 0,  9,  5,  7, 0,  9,  7,  2, 0,  9,  2,  0, 0,
		 2,  7, 11, 0,  2,  3, 11, 0,  0,  1,  8, 0,  1,  7,  8, 0,
		 1,  5,  7, 0, 11,  2,  1, 0, 11,  1,  7, 0,  7,  1,  5, 0,
		 9,  5,  8, 0,  8,  5,  7, 0, 10,  1,  3, 0, 10,  3, 11, 0,
		 5,  7,  0, 0,  5,  0,  9, 0,  7, 11,  0, 0,  1,  0, 10, 0,
		11, 10,  0, 0, 11, 10,  0, 0, 11,  0,  3, 0, 10,  5,  0, 0,
		 8,  0,  7, 0,  5,  7,  0, 0, 11, 10,  5, 0,  7, 11,  5, 0,
		10,  6,  5, 0,  0,  8,  3, 0,  5, 10,  6, 0,  9,  0,  1, 0,
		 5, 10,  6, 0,  1,  8,  3, 0,  1,  9,  8, 0,  5, 10,  6, 0,
		 1,  6,  5, 0,  2,  6,  1, 0,  1,  6,  5, 0,  1,  2,  6, 0,
		 3,  0,  8, 0,  9,  6,  5, 0,  9,  0,  6, 0,  0,  2,  6, 0,
		 5,  9,  8, 0,  5,  8,  2, 0,  5,  2,  6, 0,  3,  2,  8, 0,
		 2,  3, 11, 0, 10,  6,  5, 0, 11,  0,  8, 0, 11,  2,  0, 0,
		10,  6,  5, 0,  0,  1,  9, 0,  2,  3, 11, 0,  5, 10,  6, 0,
		 5, 10,  6, 0,  1,  9,  2, 0,  9, 11,  2, 0,  9,  8, 11, 0,
		 6,  3, 11, 0,  6,  5,  3, 0,  5,  1,  3, 0,  0,  8, 11, 0,
		 0, 11,  5, 0,  0,  5,  1, 0,  5, 11,  6, 0,  3, 11,  6, 0,
		 0,  3,  6, 0,  0,  6,  5, 0,  0,  5,  9, 0,  6,  5,  9, 0,
		 6,  9, 11, 0, 11,  9,  8, 0,  5, 10,  6, 0,  4,  7,  8, 0,
		 4,  3,  0, 0,  4,  7,  3, 0,  6,  5, 10, 0,  1,  9,  0, 0,
		 5, 10,  6, 0,  8,  4,  7, 0, 10,  6,  5, 0,  1,  9,  7, 0,
		 1,  7,  3, 0,  7,  9,  4, 0,  6,  1,  2, 0,  6,  5,  1, 0,
		 4,  7,  8, 0,  1,  2,  5, 0,  5,  2,  6, 0,  3,  0,  4, 0,
		 3,  4,  7, 0,  8,  4,  7, 0,  9,  0,  5, 0,  0,  6,  5, 0,
		 0,  2,  6, 0,  7,  3,  9, 0,  7,  9,  4, 0,  3,  2,  9, 0,
		 5,  9,  6, 0,  2,  6,  9, 0,  3, 11,  2, 0,  7,  8,  4, 0,
		10,  6,  5, 0,  5, 10,  6, 0,  4,  7,  2, 0,  4,  2,  0, 0,
		 2,  7, 11, 0,  0,  1,  9, 0,  4,  7,  8, 0,  2,  3, 11, 0,
		 5, 10,  6, 0,  9,  2,  1, 0,  9, 11,  2, 0,  9,  4, 11, 0,
		 7, 11,  4, 0,  5, 10,  6, 0,  8,  4,  7, 0,  3, 11,  5, 0,
		 3,  5,  1, 0,  5, 11,  6, 0,  5,  1, 11, 0,  5, 11,  6, 0,
		 1,  0, 11, 0,  7, 11,  4, 0,  0,  4, 11, 0,  0,  5,  9, 0,
		 0,  6,  5, 0,  0,  3,  6, 0, 11,  6,  3, 0,  8,  4,  7, 0,
		 6,  5,  9, 0,  6,  9, 11, 0,  4,  7,  9, 0,  7, 11,  9, 0,
		10,  4,  9, 0,  6,  4, 10, 0,  4, 10,  6, 0,  4,  9, 10, 0,
		 0,  8,  3, 0, 10,  0,  1, 0, 10,  6,  0, 0,  6,  4,  0, 0,
		 8,  3,  1, 0,  8,  1,  6, 0,  8,  6,  4, 0,  6,  1, 10, 0,
		 1,  4,  9, 0,  1,  2,  4, 0,  2,  6,  4, 0,  3,  0,  8, 0,
		 1,  2,  9, 0,  2,  4,  9, 0,  2,  6,  4, 0,  0,  2,  4, 0,
		 4,  2,  6, 0,  8,  3,  2, 0,  8,  2,  4, 0,  4,  2,  6, 0,
		10,  4,  9, 0, 10,  6,  4, 0, 11,  2,  3, 0,  0,  8,  2, 0,
		 2,  8, 11, 0,  4,  9, 10, 0,  4, 10,  6, 0,  3, 11,  2, 0,
		 0,  1,  6, 0,  0,  6,  4, 0,  6,  1, 10, 0,  6,  4,  1, 0,
		 6,  1, 10, 0,  4,  8,  1, 0,  2,  1, 11, 0,  8, 11,  1, 0,
		 9,  6,  4, 0,  9,  3,  6, 0,  9,  1,  3, 0, 11,  6,  3, 0,
		 8, 11,  1, 0,  8,  1,  0, 0, 11,  6,  1, 0,  9,  1,  4, 0,
		 6,  4,  1, 0,  3, 11,  6, 0,  3,  6,  0, 0,  0,  6,  4, 0,
		 6,  4,  8, 0, 11,  6,  8, 0,  7, 10,  6, 0,  7,  8, 10, 0,
		 8,  9, 10, 0,  0,  7,  3, 0,  0, 10,  7, 0,  0,  9, 10, 0,
		 6,  7, 10, 0, 10,  6,  7, 0,  1, 10,  7, 0,  1,  7,  8, 0,
		 1,  8,  0, 0, 10,  6,  7, 0, 10,  7,  1, 0,  1,  7,  3, 0,
		 1,  2,  6, 0,  1,  6,  8, 0,  1,  8,  9, 0,  8,  6,  7, 0,
		 2,  6,  9, 0,  2,  9,  1, 0,  6,  7,  9, 0,  0,  9,  3, 0,
		 7,  3,  9, 0,  7,  8,  0, 0,  7,  0,  6, 0,  6,  0,  2, 0,
		 7,  3,  2, 0,  6,  7,  2, 0,  2,  3, 11, 0, 10,  6,  8, 0,
		10,  8,  9, 0,  8,  6,  7, 0,  2,  0,  7, 0,  2,  7, 11, 0,
		 0,  9,  7, 0,  6,  7, 10, 0,  9, 10,  7, 0,  1,  8,  0, 0,
		 1,  7,  8, 0,  1, 10,  7, 0,  6,  7, 10, 0,  2,  3, 11, 0,
		11,  2,  1, 0, 11,  1,  7, 0, 10,  6,  1, 0,  6,  7,  1, 0,
		 8,  9,  6, 0,  8,  6,  7, 0,  9,  1,  6, 0, 11,  6,  3, 0,
		 1,  3,  6, 0,  0,  9,  1, 0, 11,  6,  7, 0,  7,  8,  0, 0,
		 7,  0,  6, 0,  3, 11,  0, 0, 11,  6,  0, 0,  7, 11,  6, 0,
		 7,  6, 11, 0,  3,  0,  8, 0, 11,  7,  6, 0,  0,  1,  9, 0,
		11,  7,  6, 0,  8,  1,  9, 0,  8,  3,  1, 0, 11,  7,  6, 0,
		10,  1,  2, 0,  6, 11,  7, 0,  1,  2, 10, 0,  3,  0,  8, 0,
		 6, 11,  7, 0,  2,  9,  0, 0,  2, 10,  9, 0,  6, 11,  7, 0,
		 6, 11,  7, 0,  2, 10,  3, 0, 10,  8,  3, 0, 10,  9,  8, 0,
		 7,  2,  3, 0,  6,  2,  7, 0,  7,  0,  8, 0,  7,  6,  0, 0,
		 6,  2,  0, 0,  2,  7,  6, 0,  2,  3,  7, 0,  0,  1,  9, 0,
		 1,  6,  2, 0,  1,  8,  6, 0,  1,  9,  8, 0,  8,  7,  6, 0,
		10,  7,  6, 0, 10,  1,  7, 0,  1,  3,  7, 0, 10,  7,  6, 0,
		 1,  7, 10, 0,  1,  8,  7, 0,  1,  0,  8, 0,  0,  3,  7, 0,
		 0,  7, 10, 0,  0, 10,  9, 0,  6, 10,  7, 0,  7,  6, 10, 0,
		 7, 10,  8, 0,  8, 10,  9, 0,  6,  8,  4, 0, 11,  8,  6, 0,
		 3,  6, 11, 0,  3,  0,  6, 0,  0,  4,  6, 0,  8,  6, 11, 0,
		 8,  4,  6, 0,  9,  0,  1, 0,  9,  4,  6, 0,  9,  6,  3, 0,
		 9,  3,  1, 0, 11,  3,  6, 0,  6,  8,  4, 0,  6, 11,  8, 0,
		 2, 10,  1, 0,  1,  2, 10, 0,  3,  0, 11, 0,  0,  6, 11, 0,
		 0,  4,  6, 0,  4, 11,  8, 0,  4,  6, 11, 0,  0,  2,  9, 0,
		 2, 10,  9, 0, 10,  9,  3, 0, 10,  3,  2, 0,  9,  4,  3, 0,
		11,  3,  6, 0,  4,  6,  3, 0,  8,  2,  3, 0,  8,  4,  2, 0,
		 4,  6,  2, 0,  0,  4,  2, 0,  4,  6,  2, 0,  1,  9,  0, 0,
		 2,  3,  4, 0,  2,  4,  6, 0,  4,  3,  8, 0,  1,  9,  4, 0,
		 1,  4,  2, 0,  2,  4,  6, 0,  8,  1,  3, 0,  8,  6,  1, 0,
		 8,  4,  6, 0,  6, 10,  1, 0, 10,  1,  0, 0, 10,  0,  6, 0,
		 6,  0,  4, 0,  4,  6,  3, 0,  4,  3,  8, 0,  6, 10,  3, 0,
		 0,  3,  9, 0, 10,  9,  3, 0, 10,  9,  4, 0,  6, 10,  4, 0,
		 4,  9,  5, 0,  7,  6, 11, 0,  0,  8,  3, 0,  4,  9,  5, 0,
		11,  7,  6, 0,  5,  0,  1, 0,  5,  4,  0, 0,  7,  6, 11, 0,
		11,  7,  6, 0,  8,  3,  4, 0,  3,  5,  4, 0,  3,  1,  5, 0,
		 9,  5,  4, 0, 10,  1,  2, 0,  7,  6, 11, 0,  6, 11,  7, 0,
		 1,  2, 10, 0,  0,  8,  3, 0,  4,  9,  5, 0,  7,  6, 11, 0,
		 5,  4, 10, 0,  4,  2, 10, 0,  4,  0,  2, 0,  3,  4,  8, 0,
		 3,  5,  4, 0,  3,  2,  5, 0, 10,  5,  2, 0, 11,  7,  6, 0,
		 7,  2,  3, 0,  7,  6,  2, 0,  5,  4,  9, 0,  9,  5,  4, 0,
		 0,  8,  6, 0,  0,  6,  2, 0,  6,  8,  7, 0,  3,  6,  2, 0,
		 3,  7,  6, 0,  1,  5,  0, 0,  5,  4,  0, 0,  6,  2,  8, 0,
		 6,  8,  7, 0,  2,  1,  8, 0,  4,  8,  5, 0,  1,  5,  8, 0,
		 9,  5,  4, 0, 10,  1,  6, 0,  1,  7,  6, 0,  1,  3,  7, 0,
		 1,  6, 10, 0,  1,  7,  6, 0,  1,  0,  7, 0,  8,  7,  0, 0,
		 9,  5,  4, 0,  4,  0, 10, 0,  4, 10,  5, 0,  0,  3, 10, 0,
		 6, 10,  7, 0,  3,  7, 10, 0,  7,  6, 10, 0,  7, 10,  8, 0,
		 5,  4, 10, 0,  4,  8, 10, 0,  6,  9,  5, 0,  6, 11,  9, 0,
		11,  8,  9, 0,  3,  6, 11, 0,  0,  6,  3, 0,  0,  5,  6, 0,
		 0,  9,  5, 0,  0, 11,  8, 0,  0,  5, 11, 0,  0,  1,  5, 0,
		 5,  6, 11, 0,  6, 11,  3, 0,  6,  3,  5, 0,  5,  3,  1, 0,
		 1,  2, 10, 0,  9,  5, 11, 0,  9, 11,  8, 0, 11,  5,  6, 0,
		 0, 11,  3, 0,  0,  6, 11, 0,  0,  9,  6, 0,  5,  6,  9, 0,
		 1,  2, 10, 0, 11,  8,  5, 0, 11,  5,  6, 0,  8,  0,  5, 0,
		10,  5,  2, 0,  0,  2,  5, 0,  6, 11,  3, 0,  6,  3,  5, 0,
		 2, 10,  3, 0, 10,  5,  3, 0,  5,  8,  9, 0,  5,  2,  8, 0,
		 5,  6,  2, 0,  3,  8,  2, 0,  9,  5,  6, 0,  9,  6,  0, 0,
		 0,  6,  2, 0,  1,  5,  8, 0,  1,  8,  0, 0,  5,  6,  8, 0,
		 3,  8,  2, 0,  6,  2,  8, 0,  1,  5,  6, 0,  2,  1,  6, 0,
		 1,  3,  6, 0,  1,  6, 10, 0,  3,  8,  6, 0,  5,  6,  9, 0,
		 8,  9,  6, 0, 10,  1,  0, 0, 10,  0,  6, 0,  9,  5,  0, 0,
		 5,  6,  0, 0,  0,  3,  8, 0,  5,  6, 10, 0, 10,  5,  6, 0,
		11,  5, 10, 0,  7,  5, 11, 0, 11,  5, 10, 0, 11,  7,  5, 0,
		 8,  3,  0, 0,  5, 11,  7, 0,  5, 10, 11, 0,  1,  9,  0, 0,
		10,  7,  5, 0, 10, 11,  7, 0,  9,  8,  1, 0,  8,  3,  1, 0,
		11,  1,  2, 0, 11,  7,  1, 0,  7,  5,  1, 0,  0,  8,  3, 0,
		 1,  2,  7, 0,  1,  7,  5, 0,  7,  2, 11, 0,  9,  7,  5, 0,
		 9,  2,  7, 0,  9,  0,  2, 0,  2, 11,  7, 0,  7,  5,  2, 0,
		 7,  2, 11, 0,  5,  9,  2, 0,  3,  2,  8, 0,  9,  8,  2, 0,
		 2,  5, 10, 0,  2,  3,  5, 0,  3,  7,  5, 0,  8,  2,  0, 0,
		 8,  5,  2, 0,  8,  7,  5, 0, 10,  2,  5, 0,  9,  0,  1, 0,
		 5, 10,  3, 0,  5,  3,  7, 0,  3, 10,  2, 0,  9,  8,  2, 0,
		 9,  2,  1, 0,  8,  7,  2, 0, 10,  2,  5, 0,  7,  5,  2, 0,
		 1,  3,  5, 0,  3,  7,  5, 0,  0,  8,  7, 0,  0,  7,  1, 0,
		 1,  7,  5, 0,  9,  0,  3, 0,  9,  3,  5, 0,  5,  3,  7, 0,
		 9,  8,  7, 0,  5,  9,  7, 0,  5,  8,  4, 0,  5, 10,  8, 0,
		10, 11,  8, 0,  5,  0,  4, 0,  5, 11,  0, 0,  5, 10, 11, 0,
		11,  3,  0, 0,  0,  1,  9, 0,  8,  4, 10, 0,  8, 10, 11, 0,
		10,  4,  5, 0, 10, 11,  4, 0, 10,  4,  5, 0, 11,  3,  4, 0,
		 9,  4,  1, 0,  3,  1,  4, 0,  2,  5,  1, 0,  2,  8,  5, 0,
		 2, 11,  8, 0,  4,  5,  8, 0,  0,  4, 11, 0,  0, 11,  3, 0,
		 4,  5, 11, 0,  2, 11,  1, 0,  5,  1, 11, 0,  0,  2,  5, 0,
		 0,  5,  9, 0,  2, 11,  5, 0,  4,  5,  8, 0, 11,  8,  5, 0,
		 9,  4,  5, 0,  2, 11,  3, 0,  2,  5, 10, 0,  3,  5,  2, 0,
		 3,  4,  5, 0,  3,  8,  4, 0,  5, 10,  2, 0,  5,  2,  4, 0,
		 4,  2,  0, 0,  3, 10,  2, 0,  3,  5, 10, 0,  3,  8,  5, 0,
		 4,  5,  8, 0,  0,  1,  9, 0,  5, 10,  2, 0,  5,  2,  4, 0,
		 1,  9,  2, 0,  9,  4,  2, 0,  8,  4,  5, 0,  8,  5,  3, 0,
		 3,  5,  1, 0,  0,  4,  5, 0,  1,  0,  5, 0,  8,  4,  5, 0,
		 8,  5,  3, 0,  9,  0,  5, 0,  0,  3,  5, 0,  9,  4,  5, 0,
		 4, 11,  7, 0,  4,  9, 11, 0,  9, 10, 11, 0,  0,  8,  3, 0,
		 4,  9,  7, 0,  9, 11,  7, 0,  9, 10, 11, 0,  1, 10, 11, 0,
		 1, 11,  4, 0,  1,  4,  0, 0,  7,  4, 11, 0,  3,  1,  4, 0,
		 3,  4,  8, 0,  1, 10,  4, 0,  7,  4, 11, 0, 10, 11,  4, 0,
		 4, 11,  7, 0,  9, 11,  4, 0,  9,  2, 11, 0,  9,  1,  2, 0,
		 9,  7,  4, 0,  9, 11,  7, 0,  9,  1, 11, 0,  2, 11,  1, 0,
		 0,  8,  3, 0, 11,  7,  4, 0, 11,  4,  2, 0,  2,  4,  0, 0,
		11,  7,  4, 0, 11,  4,  2, 0,  8,  3,  4, 0,  3,  2,  4, 0,
		 2,  9, 10, 0,  2,  7,  9, 0,  2,  3,  7, 0,  7,  4,  9, 0,
		 9, 10,  7, 0,  9,  7,  4, 0, 10,  2,  7, 0,  8,  7,  0, 0,
		 2,  0,  7, 0,  3,  7, 10, 0,  3, 10,  2, 0,  7,  4, 10, 0,
		 1, 10,  0, 0,  4,  0, 10, 0,  1, 10,  2, 0,  8,  7,  4, 0,
		 4,  9,  1, 0,  4,  1,  7, 0,  7,  1,  3, 0,  4,  9,  1, 0,
		 4,  1,  7, 0,  0,  8,  1, 0,  8,  7,  1, 0,  4,  0,  3, 0,
		 7,  4,  3, 0,  4,  8,  7, 0,  9, 10,  8, 0, 10, 11,  8, 0,
		 3,  0,  9, 0,  3,  9, 11, 0, 11,  9, 10, 0,  0,  1, 10, 0,
		 0, 10,  8, 0,  8, 10, 11, 0,  3,  1, 10, 0, 11,  3, 10, 0,
		 1,  2, 11, 0,  1, 11,  9, 0,  9, 11,  8, 0,  3,  0,  9, 0,
		 3,  9, 11, 0,  1,  2,  9, 0,  2, 11,  9, 0,  0,  2, 11, 0,
		 8,  0, 11, 0,  3,  2, 11, 0,  2,  3,  8, 0,  2,  8, 10, 0,
		10,  8,  9, 0,  9, 10,  2, 0,  0,  9,  2, 0,  2,  3,  8, 0,
		 2,  8, 10, 0,  0,  1,  8, 0,  1, 10,  8, 0,  1, 10,  2, 0,
		 1,  3,  8, 0,  9,  1,  8, 0,  0,  9,  1, 0,  0,  3,  8, 0
	};

	return reinterpret_cast< flink::uint4 const * >( triTable );
}



// static 
template< int BrickRes >
void svc::Mesher::Generate
(
	Volume< BrickRes > const & volume,
	Cache & cache,

	flink::vector< flink::float4 > & outVertices,
	flink::vector< unsigned > & outVertexIDs,
	flink::vector< unsigned > & outIndices
)
{
	outVertices.clear();
	outVertexIDs.clear();
	outIndices.clear();

	cache.Reset( volume.Resolution() );

	unsigned const resMinus1 = volume.Resolution() - 1;

	while( cache.NextSlice( volume.Data().keys_first(), volume.Data().values_first(), volume.Data().size() ) )
	{
		for( int i = cache.CachedRange().first; i < cache.CachedRange().second; i++ )
		{
			unsigned x0, y0, z0;
			flink::unpackInts( volume.Data().keys_first()[ i ], x0, y0, z0 );

			unsigned x1 = std::min( x0 + 1, resMinus1 );
			unsigned y1 = std::min( y0 + 1, resMinus1 );
			unsigned z1 = std::min( z0 + 1, resMinus1 );

			Voxel v[ 8 ];

			v[ 2 ] = cache.CachedSlices().first[ x0 + volume.Resolution() * y0 ];

			if( 0 == v[ 2 ].Weight() )
				continue;

			v[ 3 ] = cache.CachedSlices().first[ x1 + volume.Resolution() * y0 ];
			v[ 6 ] = cache.CachedSlices().first[ x0 + volume.Resolution() * y1 ];
			v[ 7 ] = cache.CachedSlices().first[ x1 + volume.Resolution() * y1 ];
					
			v[ 1 ] = cache.CachedSlices().second[ x0 + volume.Resolution() * y0 ];
			v[ 0 ] = cache.CachedSlices().second[ x1 + volume.Resolution() * y0 ];
			v[ 5 ] = cache.CachedSlices().second[ x0 + volume.Resolution() * y1 ];
			v[ 4 ] = cache.CachedSlices().second[ x1 + volume.Resolution() * y1 ];
			
			// Generate vertices
			float d[ 8 ];
			d[ 1 ] = v[ 1 ].Distance( volume.TruncationMargin() );
			d[ 2 ] = v[ 2 ].Distance( volume.TruncationMargin() );
			d[ 3 ] = v[ 3 ].Distance( volume.TruncationMargin() );
			d[ 6 ] = v[ 6 ].Distance( volume.TruncationMargin() );

			flink::float4 vert000 = volume.VoxelCenter( x0, y0, z0 );
			unsigned i000 = flink::packInts( x0, y0, z0 );

			// TODO: Re-evaluate interpolation

			if( v[ 3 ].Weight() > 0 && d[ 2 ] * d[ 3 ] < 0.0f )
			{
				outVertexIDs.push_back( 3 * i000 );
				outVertices.push_back( flink::float4
				(
					vert000.x + flink::lerp( 0.0f, volume.VoxelLength(), v[ 2 ].Weight() * abs( d[ 3 ] ), v[ 3 ].Weight() * abs( d[ 2 ] ) ),
					vert000.y,
					vert000.z,
					1.0f
				));
			}
				
			if( v[ 6 ].Weight() > 0 && d[ 2 ] * d[ 6 ] < 0.0f )
			{
				outVertexIDs.push_back( 3 * i000 + 1 );
				outVertices.push_back( flink::float4
				(
					vert000.x,
					vert000.y + flink::lerp( 0.0f, volume.VoxelLength(), v[ 2 ].Weight() * abs( d[ 6 ] ), v[ 6 ].Weight() * abs( d[ 2 ] ) ),
					vert000.z,
					1.0f
				));
			}
				
			if( v[ 1 ].Weight() > 0 && d[ 2 ] * d[ 1 ] < 0.0f )
			{
				outVertexIDs.push_back( 3 * i000 + 2 );
				outVertices.push_back( flink::float4
				(
					vert000.x,
					vert000.y,
					vert000.z + flink::lerp( 0.0f, volume.VoxelLength(), v[ 2 ].Weight() * abs( d[ 1 ] ), v[ 1 ].Weight() * abs( d[ 2 ] ) ),
					1.0f
				));
			}

			// Generate indices
			bool skip = false;
			for( int i = 0; i < 8; i++ )
				skip = skip || ( 0 == v[ i ].Weight() );

			if( skip ||					
				x0 == resMinus1 ||
				y0 == resMinus1 ||
				z0 == resMinus1 )
				continue;

			d[ 0 ] = v[ 0 ].Distance( volume.TruncationMargin() );
			d[ 4 ] = v[ 4 ].Distance( volume.TruncationMargin() );
			d[ 5 ] = v[ 5 ].Distance( volume.TruncationMargin() );
			d[ 7 ] = v[ 7 ].Distance( volume.TruncationMargin() );

			int lutIdx = 0;
			for( int i = 0; i < 8; i++ )
				if( d[ i ] < 0 )
					lutIdx |= ( 1u << i );

			// Maps local edge indices to global vertex indices
			unsigned localToGlobal[ 12 ];
			localToGlobal[  0 ] = flink::packInts( x0, y0, z1 ) * 3;
			localToGlobal[  1 ] = flink::packInts( x0, y0, z0 ) * 3 + 2;
			localToGlobal[  2 ] = flink::packInts( x0, y0, z0 ) * 3;
			localToGlobal[  3 ] = flink::packInts( x1, y0, z0 ) * 3 + 2;
			localToGlobal[  4 ] = flink::packInts( x0, y1, z1 ) * 3;
			localToGlobal[  5 ] = flink::packInts( x0, y1, z0 ) * 3 + 2;
			localToGlobal[  6 ] = flink::packInts( x0, y1, z0 ) * 3;
			localToGlobal[  7 ] = flink::packInts( x1, y1, z0 ) * 3 + 2;
			localToGlobal[  8 ] = flink::packInts( x1, y0, z1 ) * 3 + 1;
			localToGlobal[  9 ] = flink::packInts( x0, y0, z1 ) * 3 + 1;
			localToGlobal[ 10 ] = flink::packInts( x0, y0, z0 ) * 3 + 1;
			localToGlobal[ 11 ] = flink::packInts( x1, y0, z0 ) * 3 + 1;

			for (
				int i = TriOffsets()[ lutIdx ],
				end   = TriOffsets()[ std::min( 255, lutIdx + 1 ) ];
				i < end;
				i++
			)
			{
				flink::uint4 tri = TriTable()[ i ];
				outIndices.push_back( localToGlobal[ tri.x ] );
				outIndices.push_back( localToGlobal[ tri.y ] );
				outIndices.push_back( localToGlobal[ tri.z ] );
			}
		}
	}
}

// static 
void svc::Mesher::VertexIDsToIndices
(
	flink::vector< unsigned > const & vertexIDs,

	flink::vector< unsigned > & inOutIndices,
	flink::vector< unsigned > & tmpIndexIDs,
	flink::vector< char > & tmpScratchPad
)
{
	tmpIndexIDs.resize( inOutIndices.size() );
	for( int i = 0; i < inOutIndices.size(); i++ )
		tmpIndexIDs[ i ] = i;

	flink::radix_sort
	( 
		inOutIndices.begin(), 
		tmpIndexIDs.begin(), 
		inOutIndices.size(), 
		tmpScratchPad 
	);

	tmpScratchPad.resize( inOutIndices.size() * sizeof( unsigned ) );
	unsigned * tmp = reinterpret_cast< unsigned * >( tmpScratchPad.begin() );

	int j = 0;
	for( int i = 0; i < vertexIDs.size(); i++ )
		while( j < inOutIndices.size() && inOutIndices[ j ] == vertexIDs[ i ] )
			tmp[ j++ ] = i;

	for( int i = 0; i < inOutIndices.size(); i++ )
		inOutIndices[ tmpIndexIDs[ i ] ] = tmp[ i ];
}



template void svc::Mesher::Triangulate<1>(const Volume<1>&, Cache&, flink::vector<flink::float4>&, flink::vector<unsigned>&);
template void svc::Mesher::Triangulate<2>(const Volume<2>&, Cache&, flink::vector<flink::float4>&, flink::vector<unsigned>&);
template void svc::Mesher::Triangulate<4>(const Volume<4>&, Cache&, flink::vector<flink::float4>&, flink::vector<unsigned>&);

template void svc::Mesher::Generate<1>(const Volume<1>&, Cache&, flink::vector<flink::float4>&, flink::vector<unsigned>&, flink::vector<unsigned>&);
template void svc::Mesher::Generate<2>(const Volume<2>&, Cache&, flink::vector<flink::float4>&, flink::vector<unsigned>&, flink::vector<unsigned>&);
template void svc::Mesher::Generate<4>(const Volume<4>&, Cache&, flink::vector<flink::float4>&, flink::vector<unsigned>&, flink::vector<unsigned>&);