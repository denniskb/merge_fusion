#pragma once

#include <cstdint>
#include <vector>

#include <kifi/util/math.h>



namespace kifi {

class Volume;

class Mesher
{
public:
	void Mesh( Volume const & volume, std::vector< util::float3 > & outVertices );
	void Mesh( Volume const & volume, std::vector< util::float3 > & outVertices, std::vector< unsigned > & outIndices );

	static void Mesh2Obj
	(
		std::vector< util::float3 > const & vertices,
		std::vector< unsigned > const & indices,

		char const * outObjFileName
	);

private:
	std::vector< unsigned > m_vertexIDs;
	std::vector< unsigned > m_indexIDs;
	std::vector< unsigned > m_tmpScratchPad;

	template< bool GenerateTriangles >
	void Generate( Volume const & volume, std::vector< util::float3 > & outVertices );

	static int const * TriOffsets();
	static int const * TriTable();
};

} // namespace