#pragma once

#include <flink/math.h>
#include <flink/vector.h>



namespace svc {

class Cache;
class Volume;

class Mesher
{
public:
	/*
	Marching cubes ported from http://paulbourke.net/geometry/polygonise/
	*/
	void Triangulate
	(
		Volume const & volume,
		Cache & cache,

		flink::vector< flink::float4 > & outVertices,
		flink::vector< flink::uint4 > & outTriangles
	);

	static void Mesh2Obj
	(
		flink::vector< flink::float4 > const & vertices,
		flink::vector< flink::uint4 > const & triangles,

		char const * outObjFileName
	);

	static int const * TriOffsets();
	static flink::uint4 const * TriTable();

private:
	flink::vector< unsigned > m_vertexIDs;
	flink::vector< char > m_scratchPad;
};

}