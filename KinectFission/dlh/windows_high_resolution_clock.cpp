#ifdef _WIN32

#include <cstdint>

#include "windows_high_resolution_clock.h"
#include "windows_nominmax_strict_lean.h"



dlh::chrono::high_resolution_clock::time_point 
dlh::chrono::high_resolution_clock::now()
{
	static std::int_least64_t const freq = []()
	{
		LARGE_INTEGER freq;
		QueryPerformanceFrequency( & freq );
		return freq.QuadPart;
	}();

	LARGE_INTEGER t;
	QueryPerformanceCounter( & t );

	return time_point( duration( t.QuadPart * period::den / freq ) );
}

#endif