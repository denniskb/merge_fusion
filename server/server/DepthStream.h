/*
Reads KPPL raw depth streams version 1 as produced by poly2depth.
The file format documentation can be found in poly2depth/readme.txt.
*/

#pragma once

#include <vector>



namespace kppl {

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
	bool NextFrame( std::vector< short > & outFrame );

private:
	FILE * m_file;
	int m_nFrames;
	int m_iFrame;
};

}