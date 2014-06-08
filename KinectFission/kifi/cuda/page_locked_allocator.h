#pragma once

#include <cstddef>
#include <memory>



namespace kifi {
namespace cuda {

template< typename T >
class page_locked_allocator : public std::allocator< T >
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

#pragma warning( push )
#pragma warning( disable : 4100 ) // unreferenced parameters 'n' and 'hint'

template< typename T >
typename page_locked_allocator< T >::pointer 
page_locked_allocator< T >::allocate( std::size_t n, const_pointer hint )
{
	if( 0 == n )
		return nullptr;

	pointer result;
	cudaMallocHost( & result, n * sizeof( T ) );
	return result;
}

template< typename T >
void page_locked_allocator< T >::deallocate( pointer p, std::size_t n )
{
	if( p )
		cudaFreeHost( p );
}

#pragma warning( pop )

}} // namespace

#pragma endregion