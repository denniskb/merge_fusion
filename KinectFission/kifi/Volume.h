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

	inline util::vector VoxelCenter( util::vector index ) const;
	// TODO: Deprecated!
	inline util::float3 VoxelCenter( int x, int y, int z ) const;
	inline util::vector VoxelIndex( util::vector world ) const;

	util::flat_map< unsigned, Voxel > & Data();
	util::flat_map< unsigned, Voxel > const & Data() const;

private:
	// for VoxelCenter
	util::vector m_tmpVoxelLen;
	util::vector m_tmpVoxelLenOver2PlusMin;

	// for VoxelIndex
	util::vector m_tmpVoxelLenInv;
	util::vector m_tmpMinOverNegVoxelLen;

	// TODO: deprecated
	float m_tmpVoxelLenf;
	float m_tmpVoxelLenOver2PlusMinf;

	int m_res;
	float m_sideLen;
	float m_truncMargin;

	util::flat_map< unsigned, Voxel > m_data;
};

} // namespace



#pragma region Implementation of Inline Functions

namespace kifi {

util::vector Volume::VoxelCenter( util::vector index ) const
{
	return index * m_tmpVoxelLen + m_tmpVoxelLenOver2PlusMin;
}

util::float3 Volume::VoxelCenter( int x, int y, int z ) const
{
	util::float3 result;

	result.x = x * m_tmpVoxelLenf + m_tmpVoxelLenOver2PlusMinf;
	result.y = y * m_tmpVoxelLenf + m_tmpVoxelLenOver2PlusMinf;
	result.z = z * m_tmpVoxelLenf + m_tmpVoxelLenOver2PlusMinf;

	return result;
}

util::vector Volume::VoxelIndex( util::vector world ) const
{
	return world * m_tmpVoxelLenInv + m_tmpMinOverNegVoxelLen;
}

} // namespace

#pragma endregion