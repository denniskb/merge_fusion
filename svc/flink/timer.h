#pragma once

#include <string>
#include <vector>

#define NOMINMAX
#include <Windows.h>



namespace flink {

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