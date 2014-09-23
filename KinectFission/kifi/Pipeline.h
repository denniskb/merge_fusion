#pragma once

#include <kifi/util/vector2d.h>

#include <kifi/DepthSensorParams.h>
#include <kifi/ICP.h>
#include <kifi/Integrator.h>
#include <kifi/Mesher.h>
#include <kifi/Volume.h>



namespace kifi {

class Pipeline
{
public:
	Pipeline
	(
		DepthSensorParams const & cameraParams,

		int volumeResolution = 512, 
		float volumeSideLength = 4.0f, 
		float truncationMargin = 0.02f
	);

	void Integrate( util::vector2d< float > const & rawDepthMap, std::size_t nPoints = 10000 );
	void Integrate( util::vector2d< float > const & rawDepthMap, util::float4x4 const & eyeToWorld );

	void Mesh( std::vector< VertexPositionNormal > & outVertices );
	void Mesh( std::vector< VertexPositionNormal > & outVertices, std::vector< unsigned > & outIndices );

	util::float4x4 const & EyeToWorld() const;
	std::vector< VertexPositionNormal > const & SynthPointCloud();

private:
	kifi::Volume m_volume;
	Integrator m_integrator;
	Mesher m_mesher;
	ICP m_icp;

	DepthSensorParams m_camParams;
	util::float4x4 m_eyeToWorld;

	std::vector< VertexPositionNormal > m_tmpSynthPointCloud;
};

} // namespace