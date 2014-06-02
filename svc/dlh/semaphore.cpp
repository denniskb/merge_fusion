#include <condition_variable>
#include <mutex>

#include "semaphore.h"



dlh::semaphore::semaphore( int initialCount ) :
	m_count( initialCount )
{
}



void dlh::semaphore::signal( int const n )
{
	if( n < 1 )
		return;

	m_mutex.lock();
	m_count += n;
	m_mutex.unlock();

	if( 1 == n )
		m_cv.notify_one();
	else
		m_cv.notify_all();
}

void dlh::semaphore::wait()
{
	m_mutex.lock();

	while( m_count <= 0 )
		m_cv.wait( m_mutex );

	--m_count;

	m_mutex.unlock();
}