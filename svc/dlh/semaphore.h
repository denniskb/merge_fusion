#pragma once

#include <condition_variable>
#include <mutex>



namespace dlh {

class semaphore
{
public:
	explicit semaphore( int initialCount );

	void signal( int n = 1 );
	void wait();

private:
	int m_count;
	std::mutex m_mutex;
	std::condition_variable_any m_cv;
};

}