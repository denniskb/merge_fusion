#include <cuda_runtime.h>
#include <device_types.h>

#include <kifi/cuda/cuda_event.h>



namespace kifi {
namespace cuda {

cuda_event::cuda_event()
{
	cudaEventCreate( & m_event );
}

cuda_event::~cuda_event()
{
	cudaEventDestroy( m_event );
}



cuda_event::operator cudaEvent_t()
{
	return m_event;
}

void cuda_event::record( cudaStream_t stream )
{
	cudaEventRecord( m_event, stream );
}

void cuda_event::synchronize()
{
	cudaEventSynchronize( m_event );
}

}} // namespace