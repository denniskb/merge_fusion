#include <atomic>
#include <cassert>
#include <vector>

#include "barrier.h"



namespace kifi {
namespace util {

barrier::barrier( int nThreads ) :
	gate1( 0 ),
	gate2( 0 ),
	m_count( nThreads ),
	m_flags( nThreads )
{
	assert( nThreads > 0 );
}



void barrier::wait( int const threadid )
{
	assert( threadid >= 0 && threadid < m_flags.size() );

	if( m_flags[ threadid ] )
		phase2();
	else
		phase1();

	m_flags[ threadid ] = ! m_flags[ threadid ];
}



void barrier::phase1()
{
	int const nThreads = static_cast< int >( m_flags.size() );

	if( 0 == --m_count )
		gate2.signal( nThreads );

	gate2.wait();
}

void barrier::phase2()
{
	int const nThreads = static_cast< int >( m_flags.size() );

	if( nThreads == ++m_count )
		gate1.signal( nThreads );

	gate1.wait();
}

}} // namespace