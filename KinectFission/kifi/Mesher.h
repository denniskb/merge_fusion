#pragma once

#include <vector>

#include <kifi/util/math.h>



namespace kifi {

class Volume;

class Mesher
{
public:
	// Marching cubes ported from http://paulbourke.net/geometry/polygonise/
	void Triangulate
	(
		Volume const & volume,

		std::vector< util::float4 > & outVertices,
		std::vector< unsigned > & outIndices
	);

	static void Mesh2Obj
	(
		std::vector< util::float4 > const & vertices,
		std::vector< unsigned > const & indices,

		char const * outObjFileName
	);

	static int const * TriOffsets();
	static util::uint4 const * TriTable();

private:
	std::vector< unsigned > m_vertexIDs;
	std::vector< unsigned > m_indexIDs;
	std::vector< char > m_scratchPad;

	static void Generate
	(
		Volume const & volume,

		std::vector< util::float4 > & outVertices,
		std::vector< unsigned > & outVertexIDs,
		std::vector< unsigned > & outIndices
	);

	static void VertexIDsToIndices
	(
		std::vector< unsigned > const & vertexIDs,

		std::vector< unsigned > & inOutIndices,
		std::vector< unsigned > & tmpIndexIDs,
		std::vector< char > & tmpScratchPad
	);
};

} // namespace