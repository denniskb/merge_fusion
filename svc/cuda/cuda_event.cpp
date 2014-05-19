#include "cuda_event.h"

#include <cuda_runtime.h>
#include <device_types.h>



svcu::cuda_event::cuda_event()
{
	cudaEventCreate( & m_event );
}

svcu::cuda_event::~cuda_event()
{
	cudaEventDestroy( m_event );
}



svcu::cuda_event::operator cudaEvent_t()
{
	return m_event;
}

void svcu::cuda_event::record( cudaStream_t stream )
{
	cudaEventRecord( m_event, stream );
}

void svcu::cuda_event::synchronize()
{
	cudaEventSynchronize( m_event );
}