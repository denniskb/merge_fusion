#pragma once

#include <atomic>
#include <vector>

#include "semaphore.h"



namespace kifi {
namespace util {

class barrier
{
public:
	explicit barrier( int nThreads );

	void wait( int threadid );

private:
	semaphore gate1;
	semaphore gate2;
	std::atomic< int > m_count;
	std::vector< int > m_flags;

	void phase1();
	void phase2();
};

}} // namespace