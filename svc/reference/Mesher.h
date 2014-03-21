#pragma once

#include "flink.h"
#include "vector.h"



namespace svc {

class Volume;

class Mesher
{
public:
	void Triangulate
	(
		Volume const & volume,

		vector< unsigned > & outIndices,
		vector< flink::float4 > & outVertices
	);

	static void Mesh2Obj
	(
		vector< unsigned > const & indices,
		vector< flink::float4 > const & vertices,

		char const * outObjFileName
	);

private:
	vector< unsigned > m_cachedSlices;
	vector< unsigned > m_vertexIDs;
};

}