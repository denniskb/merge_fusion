#include "timer.h"

#include <string>
#include <utility>
#include <vector>

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <Windows.h>



svc::timer::timer()
{
	reset();
}



void svc::timer::record_time( std::string label )
{
	m_timings.push_back( std::make_pair( time(), std::move( label ) ) );

	reset();
}

void svc::timer::reset()
{
	QueryPerformanceCounter( & m_start );
}



void svc::timer::print() const
{
	double total = 0.0;
	for( auto it = m_timings.begin(); it != m_timings.end(); ++it )
	{
		printf( "%s: %.2fms\n", it->second.c_str(), it->first * 1000.0 );
		total += it->first;
	}

	printf( "total: %.2fms\n\n", total * 1000.0 );
}



double svc::timer::time()
{
	LARGE_INTEGER end, freq;
	QueryPerformanceCounter( & end );
	QueryPerformanceFrequency( & freq );

	return static_cast< double >( end.QuadPart - m_start.QuadPart ) / freq.QuadPart;
}