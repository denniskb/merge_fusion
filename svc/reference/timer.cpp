#include "timer.h"

#include <string>
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
	m_times.push_back( time() );
	m_labels.push_back( std::move( label ) );

	reset();
}

void svc::timer::reset()
{
	QueryPerformanceCounter( & m_start );
}



double svc::timer::time()
{
	LARGE_INTEGER end, freq;
	QueryPerformanceCounter( & end );
	QueryPerformanceFrequency( & freq );

	return static_cast< double >( end.QuadPart - m_start.QuadPart ) / freq.QuadPart;
}



void svc::timer::print()
{
	double total = 0.0;
	for( int i = 0; i < m_times.size(); i++ )
	{
		printf( "%s: %.2fms\n", m_labels[ i ].c_str(), m_times[ i ] * 1000.0 );
		total += m_times[ i ];
	}

	printf( "total: %.2fms\n\n", total * 1000.0 );
}