#pragma once



namespace kifi {

class Voxel
{
public:
	Voxel( unsigned data = 0 );
	operator unsigned() const;

	float Distance( float truncationMargin ) const;
	int Weight() const;

	void Update( float newDistance, float truncationMargin, int newWeight = 1 );

	bool operator==( Voxel rhs ) const;
	bool operator!=( Voxel rhs ) const;

private:
	unsigned m_data;

	static unsigned PackDistance( float distance, float truncationMargin );
	static float UnpackDistance( int distance, float truncationMargin );
};

} // namespace