#include <cassert>
#include <condition_variable>
#include <mutex>

#include "semaphore.h"



namespace kifi {
namespace util {

semaphore::semaphore( int initialCount ) :
	m_count( initialCount )
{
}



void semaphore::signal( int const n )
{
	assert( n > 0 );

	int newCount;
	m_mutex.lock();
		m_count += n;
		newCount = m_count;
	m_mutex.unlock();

	if( newCount > 1 )
		m_cv.notify_all();
	else if( newCount > 0 )
		m_cv.notify_one();
}

void semaphore::wait()
{
	m_mutex.lock();
		while( m_count <= 0 )
			m_cv.wait( m_mutex );

		--m_count;
	m_mutex.unlock();
}



void semaphore::reset( int count )
{
	m_count = count;
}

}} // namespace