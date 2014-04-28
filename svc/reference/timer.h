#pragma once

#include <string>
#include <utility>
#include <vector>

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <Windows.h>



namespace svc {

class timer
{
public:
	timer();

	void record_time( std::string label );
	void reset();

	void print() const;

private:
	LARGE_INTEGER m_start;
	std::vector< std::pair< double, std::string > > m_timings;

	double time();
};

}