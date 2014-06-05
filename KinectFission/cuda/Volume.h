#pragma once

#include "KernelVolume.h"



namespace svcu {

class Volume
{
public:
	Volume( int resolution, float sideLength, float truncationMargin );

	svcu::KernelVolume KernelVolume();
	svcu::KernelVolume const KernelVolume() const;

private:
	int m_res;
	float m_sideLen;
	float m_truncMargin;
};

}