#pragma once

#define NOMINMAX
#include <Windows.h>



namespace svc {

class Timer
{
public:
	Timer();

	/*
	Restarts the timer.
	*/
	void Reset();

	/*
	Returns the time in seconds that passed since
	the object was created.
	*/
	double Time();

private:
	LARGE_INTEGER m_start;
};

}