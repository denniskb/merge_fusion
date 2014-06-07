#pragma once

#include <string>
#include <utility>
#include <vector>

#include "chrono.h"



namespace kifi   {
namespace util   {
namespace chrono {

class stop_watch
{
public:
	stop_watch();

	float elapsed_milliseconds();
	void restart();

	void take_time( std::string label );
	void print_times();

private:
	high_resolution_clock::time_point m_start;
	std::vector< std::pair< float, std::string > > m_times;
};

}}} // namespace