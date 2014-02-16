#pragma once

#include <vector>



namespace kppl {

class DepthFrame
{
public:
	/*
	@precond width >= 0
	@precond height >= 0
	*/
	DepthFrame( int width = 0, int height = 0 );
	void Resize( int newWidth, int newHeight );

	int Width() const;
	int Height() const;
	int Resolution() const;

	/*
	Returns the depth value in meters at the specified texel.
	*/
	float operator()( int x, int y ) const;

	/*
	Returns a pointer to a continiuous block of memory,
	which represents this frame in row-major layout.
	*/
	short * data();

private:
	int m_width;
	int m_height;
	std::vector< short > m_data;
};

}