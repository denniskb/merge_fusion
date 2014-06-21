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

	inline util::float4 VoxelCenter( util::float4 index ) const;
	// TODO: Deprecated!
	inline util::vec3   VoxelCenter( int x, int y, int z ) const;
	inline util::float4 VoxelIndex( util::float4 world ) const;

	util::flat_map< unsigned, Voxel > & Data();
	util::flat_map< unsigned, Voxel > const & Data() const;

private:
	// for VoxelCenter
	util::float4 m_tmpVoxelLen;
	util::float4 m_tmpVoxelLenOver2PlusMin;

	// for VoxelIndex
	util::float4 m_tmpVoxelLenInv;
	util::float4 m_tmpNegVoxelLenInvTimesMin;

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

util::float4 Volume::VoxelCenter( util::float4 index ) const
{
	return util::fma( index, m_tmpVoxelLen, m_tmpVoxelLenOver2PlusMin );
}

util::vec3 Volume::VoxelCenter( int x, int y, int z ) const
{
	return util::vec3( (float) x, (float) y, (float) z ) * m_tmpVoxelLenf + m_tmpVoxelLenOver2PlusMinf;
}

util::float4 Volume::VoxelIndex( util::float4 world ) const
{
	return util::fma( world, m_tmpVoxelLenInv, m_tmpNegVoxelLenInvTimesMin );
}

} // namespace

#pragma endregion