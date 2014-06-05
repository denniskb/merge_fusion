#pragma once

#include <atomic>
#include <vector>

#include "semaphore.h"



namespace dlh {

class barrier
{
public:
	explicit barrier( unsigned nThreads );

	void wait( unsigned threadid );

private:
	semaphore gate1;
	semaphore gate2;
	std::atomic< unsigned > m_count;
	std::vector< unsigned > m_flags;

	void phase1();
	void phase2();
};

}