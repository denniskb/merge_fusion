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

	util::vec3 Minimum() const;
	util::vec3 Maximum() const;

	inline util::vec3 VoxelCenter( int x, int y, int z ) const;
	inline util::vec3 VoxelIndex( util::vec3 world ) const;

	util::flat_map< unsigned, Voxel > & Data();
	util::flat_map< unsigned, Voxel > const & Data() const;

public:
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
util::vec3 Volume::VoxelCenter( int x, int y, int z ) const
{
	return util::vec3( (float) x, (float) y, (float) z ) * m_tmpVoxelLen + m_tmpVoxelLenOver2PlusMin;
}

// result.w is undefined!
util::vec3 Volume::VoxelIndex( util::vec3 world ) const
{
	return world * m_tmpVoxelLenInv + m_tmpNegVoxelLenInvTimesMin;
}

} // namespace

#pragma endregion