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

	inline util::float4 VoxelCenter( util::float4 index ) const;
	inline util::float4 VoxelIndex( util::float4 world ) const;

	util::flat_map< unsigned, Voxel > & Data();
	util::flat_map< unsigned, Voxel > const & Data() const;

private:
	float m_tmpVoxelLen;
	float m_tmpVoxelLenOver2PlusMin;

	float m_tmpVoxelLenInv;
	float m_tmpNegVoxelLenInvTimesMin;

	int m_res;
	float m_sideLen;
	float m_truncMargin;

	util::flat_map< unsigned, Voxel > m_data;
};

} // namespace



#pragma region Implementation of Inline Functions

namespace kifi {

// result.w is undefined!
util::float4 Volume::VoxelCenter( util::float4 index ) const
{
	return index * m_tmpVoxelLen + m_tmpVoxelLenOver2PlusMin;
}

// result.w is undefined!
util::float4 Volume::VoxelIndex( util::float4 world ) const
{
	return world * m_tmpVoxelLenInv + m_tmpNegVoxelLenInvTimesMin;
}

} // namespace

#pragma endregion