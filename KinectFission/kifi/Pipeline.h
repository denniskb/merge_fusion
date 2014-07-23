#pragma once

#include <kifi/util/vector2d.h>

#include <kifi/DepthSensorParams.h>
#include <kifi/Integrator.h>
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
	void Integrate
	(
		util::vector2d< float > rawDepthMap,
		util::float4x4 const & worldToEye
	);

	// TODO: Replace this with direct meshing functionality
	Volume const & Volume() const;

private:
	DepthSensorParams m_camParams;
	kifi::Volume m_volume;
	Integrator m_integrator;
};

} // namespace