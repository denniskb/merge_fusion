#pragma once

#include <kifi/util/math.h>
#include <kifi/util/flat_map.h>

#include <kifi/Voxel.h>



namespace kifi {

class Volume
{
public:
	Volume( int resolution, float sideLength, float truncationMargin );

	int Resolution() const;
	float VoxelLength() const;
	float SideLength() const;
	float TruncationMargin() const;

	util::float4 Minimum() const;
	util::float4 Maximum() const;

	util::float4 VoxelCenter( int x, int y, int z ) const;
	inline util::float4 VoxelIndex( util::float4 world ) const;

	util::flat_map< unsigned, Voxel > & Data();
	util::flat_map< unsigned, Voxel > const & Data() const;

private:
	util::float4 m_tmpMin;
	util::float4 m_tmpResOverSize;

	int m_res;
	float m_sideLen;
	float m_truncMargin;

	util::flat_map< unsigned, Voxel > m_data;
};

} // namespace



#pragma region Implementation of Inline Functions

kifi::util::float4 kifi::Volume::VoxelIndex( util::float4 world ) const
{
	return (world - m_tmpMin) * m_tmpResOverSize;
}

#pragma endregion