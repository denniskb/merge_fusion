#pragma once

#include <cstdint>
#include <vector>

#include <kifi/util/math.h>



namespace kifi {

class Volume;

class Mesher
{
public:
	void Mesh( Volume const & volume, std::vector< util::vec3 > & outVertices );
	void Mesh( Volume const & volume, std::vector< util::vec3 > & outVertices, std::vector< std::uint32_t > & outIndices );

	static void Mesh2Obj
	(
		std::vector< util::vec3 > const & vertices,
		std::vector< unsigned > const & indices,

		char const * outObjFileName
	);

private:
	std::vector< std::uint32_t > m_vertexIDs;
	std::vector< std::uint32_t > m_indexIDs;
	std::vector< std::uint32_t > m_tmpScratchPad;

	template< bool GenerateTriangles >
	void Generate( Volume const & volume, std::vector< util::vec3 > & outVertices );

	static std::uint32_t const * TriOffsets();
	static util::uint4 const * TriTable();
};

} // namespace