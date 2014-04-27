/*
Reads flink raw depth streams version 1 as produced by poly2depth.
The file format documentation can be found in poly2depth/readme.txt.
*/

#pragma once

#include <fstream>
#include <string>

#include "dxmath.h"
#include "vector2d.h"



namespace svc {

class DepthStream
{
public:
	/*
	Opens the depth stream specified by 'file_name' and keeps it
	open until the object is destroyed.
	*/
	explicit DepthStream( std::string const & fileName );

	/*
	Copies the next frame into 'outFrame'.
	Returns true if a frame was copied, false if no more frames remain.
	*/
	bool NextFrame
	(
		vector2d< float > & outFrame,
		float4x4 & outView
	);

private:
	enum TexelType{ SHORT, FLOAT };

	std::ifstream m_file;
	
	TexelType m_texelType;
	
	int m_frameWidth;
	int m_frameHeight;

	int m_nFrames;
	int m_iFrame;

	vector2d< short > m_bufferedDepth;

	/*
	Copy and assignment are forbidden since we hold an open file handle.
	Hence, the main resource this class wraps would be unavailable anyway.
	*/
	DepthStream( DepthStream const & copy );
	DepthStream & operator=( DepthStream rhs );
};

}