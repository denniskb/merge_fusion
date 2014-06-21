#include <chrono>
#include <cstdio>
#include <ratio>
#include <string>
#include <utility>
#include <vector>

#include "stop_watch.h"



namespace kifi   {
namespace util   {
namespace chrono {

stop_watch::stop_watch()
{
	restart();
}



float stop_watch::elapsed_milliseconds()
{
	auto end = high_resolution_clock::now();

	return std::chrono::duration_cast< std::chrono::duration< float, std::milli > >( end - m_start ).count();
}

void stop_watch::restart()
{
	m_start = high_resolution_clock::now();
}



void stop_watch::take_time( std::string label )
{
	float time = elapsed_milliseconds();

	m_times.push_back( std::make_pair( time, label ) );

	restart();
}

void stop_watch::print_times()
{
	double total = 0.0;

	for( auto it = m_times.cbegin(); it != m_times.cend(); ++it )
	{
		std::printf( "%s: %.2fms\n", it->second.c_str(), it->first );
		total += it->first;
	}

	if( m_times.size() > 1 )
		std::printf( "total: %.2fms\n", total );
}

}}} // namespace