/*
A voxel as used by Volume.
Consists of distance and weight.
The whole volume defines a signed distance field.
*/

#pragma once



namespace kppl {

#define VOXEL_DIST_BITS 12
#define VOXEL_WEIGHT_BITS 4

#define VOXEL_MAX_DIST ( ( 1u << VOXEL_DIST_BITS ) - 1 )
#define VOXEL_MAX_WEIGHT ( ( 1u << VOXEL_WEIGHT_BITS ) - 1 )

#define VOXEL_MAX_DIST_OVER_2 ( VOXEL_MAX_DIST * 0.5f )

/*
newDistance and truncationMargin are provided in meters.
Integration only happens up to surface plus truncation margin.
All measurements beyond truncationMargin in either direction of the surface are clamped to truncationMargin.
*/
class Voxel
{
public:
	Voxel();

	float Distance( float truncationMargin ) const; // in meters
	int Weight() const;

	void Update( float newDistance, float truncationMargin );

	/*
	Returns true if this and rhs are identical on the bit-level.
	Please note that this comparison is only meaningful if
	both voxels were only ever updated using the same truncation margin.
	It is the programmer's responsibility to assert this.
	*/
	bool operator==( Voxel const & rhs ) const;
	bool operator!=( Voxel const & rhs ) const;

private:
	unsigned short m_distance : VOXEL_DIST_BITS;
	unsigned short m_weight : VOXEL_WEIGHT_BITS;

	/*
	Maps distance from [-truncMargin, truncMargin] to [0, 63]
	*/
	static int PackDistance( float distance, float truncationMargin );
	/*
	Maps distance from [0, 63] to [-truncMargin, truncMargin]
	*/
	static float UnpackDistance( int distance, float truncationMargin );

	static int Clamp( int x, int min, int max );
	static float Clamp( float x, float min, float max );
};

}