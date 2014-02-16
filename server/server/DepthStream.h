/*
Reads KPPL raw depth streams version 1 as produced by poly2depth.
The file format documentation can be found in poly2depth/readme.txt.
*/

#pragma once

#include <vector>

#include "flink.h"



namespace kppl {

using namespace flink;

class DepthFrame;

class DepthStream
{
public:
	/*
	Opens the depth stream specified by 'file_name' and keeps it
	open until the object is destroyed.
	*/
	DepthStream( char const * fileName );
	~DepthStream();

	/*
	Copies the next frame into 'outFrame'.
	Returns true if a frame was copied, false if no more frames remain.
	*/
	bool NextFrame
	(
		DepthFrame & outFrame,
		float4x4 & outView
	);

private:
	FILE * m_file;
	int m_nFrames;
	int m_iFrame;

	/*
	Copy and assignment are forbidden since we hold an open file handle.
	Hence, the main resource this class wraps would be unavailable anyway.
	*/
	DepthStream( DepthStream const & copy );
	DepthStream & operator=( DepthStream rhs );
};

}