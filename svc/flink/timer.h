#pragma once

#define NOMINMAX
#include <Windows.h>



namespace flink {

class timer
{
public:
	timer();

	/*
	Restarts the timer.
	*/
	void reset();

	/*
	Returns the time in seconds that passed since
	the object was created.
	*/
	double time();

private:
	LARGE_INTEGER m_start;
};

}