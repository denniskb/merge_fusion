#pragma once

#include <fstream>
#include <string>

#include <kifi/util/DirectXMathExt.h>
#include <kifi/util/vector2d.h>



namespace kifi {

class DepthStream
{
public:
	explicit DepthStream( std::string const & fileName );

	bool NextFrame
	(
		util::vector2d< float > & outFrame,
		util::float4x4 & outView
	);

private:
	enum TexelType{ SHORT, FLOAT };

	std::ifstream m_file;
	
	TexelType m_texelType;
	
	int m_frameWidth;
	int m_frameHeight;

	int m_nFrames;
	int m_iFrame;

	util::vector2d< short > m_bufferedDepth;

	DepthStream( DepthStream const & copy );
	DepthStream & operator=( DepthStream rhs );
};

} // namespace