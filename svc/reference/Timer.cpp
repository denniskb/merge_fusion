#include "Timer.h"



svc::Timer::Timer()
{
	Reset();
}



void svc::Timer::Reset()
{
	QueryPerformanceCounter( & m_start );
}



double svc::Timer::Time()
{
	LARGE_INTEGER end, freq;
	QueryPerformanceCounter( & end );
	QueryPerformanceFrequency( & freq );

	return static_cast< double >( end.QuadPart - m_start.QuadPart ) / freq.QuadPart;
}