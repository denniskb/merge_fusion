#pragma once

#include <vector>



namespace svc {

class DepthFrame
{
public:
	/*
	@precond width >= 0
	@precond height >= 0
	*/
	explicit DepthFrame( int width = 0, int height = 0 );
	void Resize( int newWidth, int newHeight );

	int Width() const;
	int Height() const;
	int Resolution() const;

	/*
	Returns the depth value in meters at the specified texel.
	*/
	float operator()( int x, int y ) const;
	float & operator()( int x, int y );

private:
	std::vector< float > m_data;
	int m_width;
	int m_height;

	static int Index2Dto1D( int x, int y, int width );
};

}