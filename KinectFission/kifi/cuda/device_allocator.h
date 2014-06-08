#pragma once

#include <cstddef>
#include <memory>



namespace kifi {
namespace cuda {

template< typename T >
class device_allocator : public std::allocator< T >
{
public:
	pointer allocate( std::size_t n, const_pointer hint = 0 );
	void deallocate( pointer p, std::size_t n );
};

}} // namespace



#pragma region Implementation

#include <cuda_runtime.h>



namespace kifi {
namespace cuda {

// unreferenced parameters 'n' and 'hint'
#pragma warning( push )
#pragma warning( disable : 4100 )

template< typename T >
typename device_allocator< T >::pointer device_allocator< T >::allocate( std::size_t n, const_pointer hint )
{
	if( 0 == n )
		return nullptr;

	pointer result;
	cudaMalloc( & result, n * sizeof( T ) );
	return result;
}

template< typename T >
void device_allocator< T >::deallocate( pointer p, std::size_t n )
{
	if( p )
		cudaFree( p );
}

#pragma warning( pop )

}} // namespace

#pragma endregion