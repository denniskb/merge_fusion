#pragma once

#include <string>
#include <utility>
#include <vector>

#include <driver_types.h>



namespace svcu {

class cuda_timer
{
public:
	cuda_timer( cudaStream_t stream = 0 );
	~cuda_timer();
	
	void record_time( std::string label );
	void reset();

	void print() const;

private:
	cudaEvent_t m_start;
	cudaEvent_t m_end;
	cudaStream_t m_stream;

	std::vector< std::pair< float, std::string > > m_timings;

	float time();
};

}