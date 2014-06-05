#pragma once

#include <driver_types.h>



namespace svcu {

class cuda_event
{
public:
	cuda_event();
	~cuda_event();

	operator cudaEvent_t();
	void record( cudaStream_t stream = 0 );
	void synchronize();

private:
	cudaEvent_t m_event;

	cuda_event( cuda_event const & );
	cuda_event & operator=( cuda_event );
};

}