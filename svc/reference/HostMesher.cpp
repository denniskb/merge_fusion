#include "HostMesher.h"

#include "HostVolume.h"



//void svc::HostMesher::Triangulate
//(
//	HostVolume const & volume,
//
//	vector< unsigned > & outIndices,
//	vector< flink::float4 > & outVertices
//)
//{
//
//}



// static 
void svc::HostMesher::Mesh2Obj
(
	vector< unsigned > const & indices,
	vector< flink::float4 > const & vertices,

	char const * outObjFileName
)
{
	FILE * file;
	fopen_s( & file, outObjFileName, "w" );

	for( int i = 0; i < vertices.size(); i++ )
	{
		auto vertex = vertices[ i ];
		fprintf_s( file, "v %f %f %f\n", vertex.x, vertex.y, vertex.z );
	}

	for( int i = 0; i < indices.size(); i += 3 )
		fprintf_s( file, "f %d %d %d\n", indices[ i ] + 1, indices[ i + 1 ] + 1, indices[ i + 2 ] + 1 );

	fclose( file );
}