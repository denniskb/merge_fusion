#include <atomic>
#include <vector>

#include "barrier.h"



dlh::barrier::barrier( unsigned nThreads ) :
	gate1( 0 ),
	gate2( 0 ),
	m_count( nThreads ),
	m_flags( nThreads )
{
}



void dlh::barrier::wait( unsigned const threadid )
{
	if( m_flags[ threadid ] )
		phase2();
	else
		phase1();

	m_flags[ threadid ] = ! m_flags[ threadid ];
}



void dlh::barrier::phase1()
{
	unsigned const nThreads = static_cast< int >( m_flags.size() );

	if( 0 == --m_count )
		gate2.signal( nThreads );

	gate2.wait();
}

void dlh::barrier::phase2()
{
	unsigned const nThreads = static_cast< int >( m_flags.size() );

	if( nThreads == ++m_count )
		gate1.signal( nThreads );

	gate1.wait();
}