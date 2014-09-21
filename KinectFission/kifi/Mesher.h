#pragma once

#include <cstdint>
#include <vector>

#include <kifi/util/math.h>



namespace kifi {

class Volume;

struct VertexPositionNormal
{
	util::float3 position;
	util::float3 normal;

	VertexPositionNormal();
	VertexPositionNormal( util::float3 position, util::float3 normal );
};



class Mesher
{
public:
    // Potentially generate NaN-Normals !
	void Mesh( Volume const & volume, std::vector< VertexPositionNormal > & outVertices );
	void Mesh( Volume const & volume, std::vector< VertexPositionNormal > & outVertices, std::vector< unsigned > & outIndices );

	static void Mesh2Obj
	(
		std::vector< VertexPositionNormal > const & vertices,
		std::vector< unsigned > const & indices,

		char const * outObjFileName
	);

private:
	std::vector< unsigned > m_vertexIDs;
	std::vector< unsigned > m_indexIDs;
	std::vector< unsigned > m_tmpScratchPad;
	std::vector< util::float3 > m_tmpNormals;

	template< bool GenerateTriangles >
	void Generate( Volume const & volume, std::vector< VertexPositionNormal > & outVertices );

	static int const * TriOffsets();
	static int const * TriTable();
};

} // namespace