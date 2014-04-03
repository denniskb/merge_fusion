#pragma once

#include <vector>



namespace svc {

class DepthFrame
{
public:
	DepthFrame();
	DepthFrame( int width, int height );
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