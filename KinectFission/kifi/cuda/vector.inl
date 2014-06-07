#include <cuda_runtime.h>

#include <kifi/cuda/cuda_event.h>



namespace kifi {
namespace cuda {

template< typename T >
vector< T >::vector( size_t initialCapacity ) :
	m_data( initialCapacity ),
	m_size( 4 ),
	m_tmpSize( 4 )
{
	clear();	
}



template< typename T >
void vector< T >::clear( cudaStream_t stream )
{
	resize( 0, stream );
}

template< typename T >
size_t vector< T >::capacity() const
{
	return m_data.capacity();
}



template< typename T >
T * vector< T >::data()
{
	return m_data.data();
}

template< typename T >
T const * vector< T >::data() const
{
	return m_data.data();
}



template< typename T >
unsigned * vector< T >::size()
{
	return m_size.data();
}

template< typename T >
unsigned const * vector< T >::size() const
{
	return m_size.data();
}



template< typename T >
cuda::kernel_vector< T > vector< T >::kernel_vector()
{
	return cuda::kernel_vector< T >( data(), size() );
}

template< typename T >
cuda::kernel_vector< T > const vector< T >::kernel_vector() const
{
	return cuda::kernel_vector< T >( data(), size() );
}



template< typename T >
void vector< T >::reserve( size_t newCapacity )
{
	if( newCapacity > capacity() )
	{
		buffer< T > newData( newCapacity );

		cudaMemcpy( newData.data(), data(), capacity() * sizeof( T ), cudaMemcpyDeviceToDevice );

		m_data.swap( newData );
	}
}

template< typename T >
void vector< T >::resize( size_t newSize, cudaStream_t stream )
{
	* m_tmpSize.data() = (unsigned) newSize;
	cudaMemcpyAsync( size(), m_tmpSize.data(), 4, cudaMemcpyHostToDevice, stream );

	reserve( newSize );
}



template< typename T >
void copy( vector< T > & dst, vector< T > const & src, cudaStream_t stream )
{
	unsigned srcSize;
	cuda_event sizeCopied;

	cudaMemcpyAsync( & srcSize, src.m_size, 4, cudaMemcpyDeviceToHost, stream );
	sizeCopied.record( stream );

	cudaMemcpyAsync( dst.m_size, src.m_size, 4, cudaMemcpyDeviceToDevice, stream );

	sizeCopied.synchronize();
	dst.reserve( srcSize );

	cudaMemcpyAsync( dst.data(), src.data(), srcSize * sizeof( T ), cudaMemcpyDeviceToDevice, stream );
}

template< typename T >
void copy( std::vector< T > & dst, vector< T > const & src, cudaStream_t stream )
{
	unsigned srcSize;
	cuda_event sizeCopied;

	cudaMemcpyAsync( & srcSize, src.size(), 4, cudaMemcpyDeviceToHost, stream );
	sizeCopied.record( stream );
	sizeCopied.synchronize();

	dst.resize( srcSize );
	cudaMemcpyAsync( dst.data(), src.data(), srcSize * sizeof( T ), cudaMemcpyDeviceToHost, stream );
}

template< typename T >
void copy( vector< T > & dst, std::vector< T > const & src, cudaStream_t stream )
{
	dst.resize( src.size(), stream );

	cudaMemcpyAsync( dst.data(), src.data(), src.size() * sizeof( T ), cudaMemcpyHostToDevice, stream );
}

}} // namespace