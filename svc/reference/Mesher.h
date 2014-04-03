#pragma once

#include <flink/math.h>
#include <flink/vector.h>

#include "Volume.h"



namespace svc {

class Cache;

class Mesher
{
public:
	/*
	Marching cubes ported from http://paulbourke.net/geometry/polygonise/
	*/
	template< int BrickRes >
	void Triangulate
	(
		Volume< BrickRes > const & volume,
		Cache & cache,

		flink::vector< flink::float4 > & outVertices,
		flink::vector< unsigned > & outIndices
	);

	static void Mesh2Obj
	(
		flink::vector< flink::float4 > const & vertices,
		flink::vector< unsigned > const & indices,

		char const * outObjFileName
	);

	static int const * TriOffsets();
	static flink::uint4 const * TriTable();

private:
	flink::vector< unsigned > m_vertexIDs;
	flink::vector< unsigned > m_indexIDs;
	flink::vector< char > m_scratchPad;

	template< int BrickRes >
	static void Generate
	(
		Volume< BrickRes > const & volume,
		Cache & cache,

		flink::vector< flink::float4 > & outVertices,
		flink::vector< unsigned > & outVertexIDs,
		flink::vector< unsigned > & outIndices
	);

	static void VertexIDsToIndices
	(
		flink::vector< unsigned > const & vertexIDs,

		flink::vector< unsigned > & inOutIndices,
		flink::vector< unsigned > & tmpIndexIDs,
		flink::vector< char > & tmpScratchPad
	);
};

}