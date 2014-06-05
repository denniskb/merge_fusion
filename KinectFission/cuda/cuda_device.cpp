#include "cuda_device.h"

#include <algorithm>

#include <cuda_runtime.h>



// static 
svcu::cuda_device svcu::cuda_device::from_id( int id )
{
	return cuda_device( id );
}

// static 
svcu::cuda_device & svcu::cuda_device::main()
{
	static cuda_device main_device( 0 );

	return main_device;
}



int svcu::cuda_device::id() const
{
	return m_id;
}

int svcu::cuda_device::max_residing_blocks
(
	int threadsPerBlock, 
	int regsPerThread, 
	int sharedMemPerBlock
) const
{
	int warpsPerBlock = ( threadsPerBlock + m_props.warpSize - 1 ) / m_props.warpSize;
	int actualThreadsPerBlock = warpsPerBlock * m_props.warpSize;

	int actualRegsPerThread = std::max( 32, ( regsPerThread + 7 ) / 8 * 8 );

	// max number of blocks per MPC, as limited by warps/regs/smem respectively

	int maxBlocksByWarps = 
		m_props.maxThreadsPerMultiProcessor / actualThreadsPerBlock;

	int maxBlocksByRegs =
		m_props.regsPerBlock / ( actualThreadsPerBlock * actualRegsPerThread );

	// m_props.sharedMemPerBlock <= 64K for the near future
	int maxBlocksBySharedMem =
		(int) m_props.sharedMemPerBlock / std::max( 1, sharedMemPerBlock );

	return std::min( maxBlocksByWarps, std::min( maxBlocksByRegs, maxBlocksBySharedMem ) ) *
		m_props.multiProcessorCount;
}

cudaDeviceProp const & svcu::cuda_device::props() const
{
	return m_props;
}
	
	
	
	
svcu::cuda_device::cuda_device( int id ) :
	m_id( id )
{
	cudaGetDeviceProperties( & m_props, id );
}