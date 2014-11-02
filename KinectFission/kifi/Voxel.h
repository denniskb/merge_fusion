#pragma once

#include <cassert>



namespace kifi {

struct Voxel
{
	inline Voxel();

	inline float Distance() const;
	inline float SafeDistance() const;
	inline float Weight() const;

	inline void Update( float newDistance, float weight = 1.0f );

private:
	float m_distance;
	float m_weight;
};

} // namespace



#pragma region Implementation

namespace kifi {

Voxel::Voxel() :
	m_distance( 0.0f ),
	m_weight( 0.0f )
{
}



float Voxel::Distance() const
{
	assert( m_weight > 0.0f );

	return m_distance / m_weight;
}

float Voxel::SafeDistance() const
{
	return m_distance / (m_weight + std::numeric_limits< float >::min());
}

float Voxel::Weight() const
{
	return m_weight;
}



void Voxel::Update( float distance, float weight )
{
	m_distance += weight * distance;
	m_weight   += weight;
}

} // namespace

#pragma endregion