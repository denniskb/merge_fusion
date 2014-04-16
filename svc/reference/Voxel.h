#pragma once



namespace svc {

/*
A single cell as used inside an voxel volume,
capable of storing a weight and a signed distance.
The entire volume thus defines a signed distance field.
All parameters are in meters.
Measurements beyond truncationMargin in either direction are clamped.

This class is managed (.m), meaning it can be used by both host and device.
*/
class Voxel
{
public:
	explicit Voxel( unsigned data = 0 );

	float Distance( float truncationMargin ) const;
	int Weight() const;

	void Update( float newDistance, float truncationMargin, int newWeight = 1 );

	bool operator==( Voxel rhs ) const;
	bool operator!=( Voxel rhs ) const;

private:
	unsigned m_data;

	/*
	Maps distance from [-truncMargin, truncMargin] to [0, 63]
	*/
	static unsigned PackDistance( float distance, float truncationMargin );
	
	/*
	Maps distance from [0, 63] to [-truncMargin, truncMargin]
	*/
	static float UnpackDistance( int distance, float truncationMargin );
};

}