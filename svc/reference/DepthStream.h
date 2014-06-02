#pragma once

#include <fstream>
#include <string>

#include <dlh/DirectXMathExt.h>
#include <dlh/vector2d.h>



namespace svc {

class DepthStream
{
public:
	explicit DepthStream( std::string const & fileName );

	bool NextFrame
	(
		dlh::vector2d< float > & outFrame,
		dlh::float4x4 & outView
	);

private:
	enum TexelType{ SHORT, FLOAT };

	std::ifstream m_file;
	
	TexelType m_texelType;
	
	int m_frameWidth;
	int m_frameHeight;

	int m_nFrames;
	int m_iFrame;

	dlh::vector2d< short > m_bufferedDepth;

	DepthStream( DepthStream const & copy );
	DepthStream & operator=( DepthStream rhs );
};

}