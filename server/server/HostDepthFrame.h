#pragma once

#include <vector>



namespace kppl {

class HostDepthFrame
{
public:
	/*
	@precond width >= 0
	@precond height >= 0
	*/
	HostDepthFrame( int width = 0, int height = 0 );
	void Resize( int newWidth, int newHeight );

	int Width() const;
	int Height() const;
	int Resolution() const;

	/*
	Returns the depth value in meters at the specified texel.
	*/
	float & operator()( int x, int y );
	float const & operator()( int x, int y ) const;

private:
	int m_width;
	int m_height;
	std::vector< float > m_data;

	static int Index2Dto1D( int x, int y, int width );
};

}