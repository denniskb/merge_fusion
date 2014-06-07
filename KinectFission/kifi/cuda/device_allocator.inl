#include <cuda_runtime.h>



namespace kifi {
namespace cuda {

#pragma warning( push )
#pragma warning( disable : 4100 ) // unreferenced parameters 'n' and 'hint'

template< typename T >
typename device_allocator< T >::pointer device_allocator< T >::allocate( size_t n, const_pointer hint )
{
	if( 0 == n )
		return nullptr;

	pointer result;
	cudaMalloc( & result, n * sizeof( T ) );
	return result;
}

template< typename T >
void device_allocator< T >::deallocate( pointer p, size_t n )
{
	if( p )
		cudaFree( p );
}

#pragma warning( pop )

}} // namespace