#pragma once

#include <kifi/util/math.h>



namespace kifi {

struct Voxel
{
	Voxel() : m_d( 0.0f ), m_w( 0 ) {}

	float Distance() const { return m_d / m_w; }
	float Weight() const { return m_w; }

	void Update( float newDistance )
	{
		m_d += newDistance;
		m_w ++;
	}

private:
	float m_d;
	float m_w;
};

} // namespace