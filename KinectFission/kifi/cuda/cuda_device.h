#pragma once

#include <driver_types.h>



namespace kifi {
namespace cuda {

class cuda_device
{
public:
	static cuda_device from_id( int id );
	static cuda_device & main();

	int id() const;
	int max_residing_blocks
	( 
		int threadsPerBlock, 
		int regsPerThread, 
		int sharedMemPerBlock = 0
	) const;
	cudaDeviceProp const & props() const;

private:
	int m_id;
	cudaDeviceProp m_props;

	explicit cuda_device( int id );
};

}} // namespace