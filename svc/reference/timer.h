#pragma once

#include <string>
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

	double time();

	void print();

private:
	LARGE_INTEGER m_start;

	std::vector< double > m_times;
	std::vector< std::string > m_labels;
};

}