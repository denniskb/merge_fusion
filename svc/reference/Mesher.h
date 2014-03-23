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

		flink::vector< unsigned > & outIndices,
		flink::vector< flink::float4 > & outVertices
	);

	static void Mesh2Obj
	(
		flink::vector< unsigned > const & indices,
		flink::vector< flink::float4 > const & vertices,

		char const * outObjFileName
	);

private:
	static int const * TriTable();

	flink::vector< unsigned > m_vertexIDs;
};

}