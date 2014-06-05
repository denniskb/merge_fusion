#pragma once

#include <string>
#include <utility>
#include <vector>

#include "cuda_event.h"



namespace svcu {

class timer
{
public:
	timer( cudaStream_t stream = 0 );
	
	float record_time( std::string label );
	void reset();

	void print() const;

private:
	cuda_event m_start;
	cuda_event m_end;
	cudaStream_t m_stream;

	std::vector< std::pair< float, std::string > > m_timings;

	float time();
};

}