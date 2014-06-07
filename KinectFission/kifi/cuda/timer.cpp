#include <string>
#include <utility>
#include <vector>

#include <cuda_runtime.h>
#include <driver_types.h>

#include <kifi/cuda/timer.h>



namespace kifi {
namespace cuda {

timer::timer( cudaStream_t stream ) :
	m_stream( stream )
{
	reset();
}
	


float timer::record_time( std::string label )
{
	float result = time();

	m_timings.push_back( std::make_pair( result, std::move( label ) ) );
	
	return result;
}

void timer::reset()
{
	m_start.record( m_stream );
}



void timer::print() const
{
	float total = 0.0f;
	
	for( auto it = m_timings.begin(); it != m_timings.end(); ++it )
	{
		printf( "%s: %.2fms\n", it->second.c_str(), it->first );
		total += it->first;
	}

	printf( "total: %.2fms\n\n", total );
}



float timer::time()
{
	m_end.record( m_stream );
	m_end.synchronize();

	float result;
	cudaEventElapsedTime( & result, m_start, m_end );
	return result;
}

}} // namepspace