#include "timer.h"



flink::timer::timer()
{
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