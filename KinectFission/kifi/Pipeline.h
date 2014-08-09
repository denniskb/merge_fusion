#pragma once

#include <kifi/util/vector2d.h>

#include <kifi/DepthSensorParams.h>
#include <kifi/ICP.h>
#include <kifi/Integrator.h>
#include <kifi/Mesher.h>
#include <kifi/Renderer.h>
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

	// TODO: Compute worldToEye via ICP instead of asking for it.
	// TODO: Overload for vector2d< short >
	// TODO: output synth depth as by-product
	void Integrate
	(
		util::vector2d< float > rawDepthMap,
		util::float4x4 const & worldToEye
	);

	void Mesh( std::vector< VertexPositionNormal > & outVertices );
	void Mesh( std::vector< VertexPositionNormal > & outVertices, std::vector< unsigned > & outIndices );

	Volume const & Volume() const;
	util::float4x4 const & EyeToWorld() const;

//private:
	DepthSensorParams m_camParams;
	kifi::Volume m_volume;
	ICP m_icp;
	Integrator m_integrator;
	Mesher m_mesher;
	Renderer m_renderer;

	int m_iFrame;
	util::float4x4 m_eyeToWorld; // or eyeToWorld???

	std::vector< util::float3 > m_tmpSynthPointCloud;
	util::vector2d< util::float3 > m_tmpSynthPointBuffer;
};

} // namespace