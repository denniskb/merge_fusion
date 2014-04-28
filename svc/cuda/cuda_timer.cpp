#include "cuda_timer.h"

#include <string>
#include <utility>
#include <vector>

#include <cuda_runtime.h>



svcu::cuda_timer::cuda_timer( cudaStream_t stream ) :
	m_stream( stream )
{
	cudaEventCreate( & m_start );
	cudaEventCreate( & m_end );

	reset();
}

svcu::cuda_timer::~cuda_timer()
{
	cudaEventDestroy( m_start );
	cudaEventDestroy( m_end );
}
	


void svcu::cuda_timer::record_time( std::string label )
{
	m_timings.push_back( std::make_pair( time(), std::move( label ) ) );
}

void svcu::cuda_timer::reset()
{
	cudaEventRecord( m_start );
}



void svcu::cuda_timer::print() const
{
	float total = 0.0f;
	
	for( auto it = m_timings.begin(); it != m_timings.end(); ++it )
	{
		printf( "%s: %.2fms\n", it->second.c_str(), it->first );
		total += it->first;
	}

	printf( "total: %.2fms\n\n", total );
}



float svcu::cuda_timer::time()
{
	cudaEventRecord( m_end, m_stream );
	cudaEventSynchronize( m_end );

	float result;
	cudaEventElapsedTime( & result, m_start, m_end );
	return result;
}