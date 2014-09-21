#pragma once

#include <cassert>



namespace kifi {

struct Voxel
{
	inline Voxel();

	inline float Distance() const;
	inline float Weight() const;

	inline void Update( float newDistance );

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

float Voxel::Weight() const
{
	return m_weight;
}



void Voxel::Update( float distance )
{
	m_distance += distance;
	m_weight++;
}

} // namespace

#pragma endregion