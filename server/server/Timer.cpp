#include "Timer.h"



kppl::Timer::Timer()
{
	Reset();
}



void kppl::Timer::Reset()
{
	QueryPerformanceCounter( & m_start );
}



double kppl::Timer::Time()
{
	LARGE_INTEGER end, freq;
	QueryPerformanceCounter( & end );
	QueryPerformanceFrequency( & freq );

	return static_cast< double >( end.QuadPart - m_start.QuadPart ) / freq.QuadPart;
}