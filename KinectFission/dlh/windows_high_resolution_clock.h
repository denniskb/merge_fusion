#pragma once

#include <chrono>
#include <cstdint>
#include <ratio>



namespace dlh { namespace chrono {

class high_resolution_clock
{
public:
	typedef std::int_least64_t								 rep;
	typedef std::nano										 period;
	typedef std::chrono::duration< rep, period >			 duration;
	typedef std::chrono::time_point< high_resolution_clock > time_point;
	
	static bool const is_steady = true;

	static time_point now();
};

}}