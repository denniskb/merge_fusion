#include "timer.h"



flink::timer::timer()
{
	reset();
}



void flink::timer::record_time( std::string label )
{
	m_times.push_back( time() );
	m_labels.push_back( std::move( label ) );

	reset();
}

void flink::timer::reset()
{
	QueryPerformanceCounter( & m_start );
}



double flink::timer::time()
{
	LARGE_INTEGER end, freq;
	QueryPerformanceCounter( & end );
	QueryPerformanceFrequency( & freq );

	return static_cast< double >( end.QuadPart - m_start.QuadPart ) / freq.QuadPart;
}



void flink::timer::print()
{
	double total = 0.0;
	for( int i = 0; i < m_times.size(); i++ )
	{
		printf( "%s: %.2fms\n", m_labels[ i ], m_times[ i ] * 1000.0 );
		total += m_times[ i ];
	}

	printf( "total: %.2fms\n\n", total * 1000.0 );
}