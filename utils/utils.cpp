#include "StdAfx.h"
#include "utils.h"
#include "asserts.h"

#ifdef LINUX
#	include <time.h>
#	include <sys/types.h>
#	include <unistd.h>
#else
#	include <windows.h>
#endif


namespace utils
{
#ifdef LINUX

void sleep_ms(int ms)
{
	struct timespec tim, tim2;
	tim.tv_sec = 0;
	tim.tv_nsec = ms * 1000000; // 10 ms
	nanosleep(&tim , &tim2);
}

bool get_binary_name(string& binary)
{
	const size_t buffer_size = 1024;
	char buffer[buffer_size];

	int res = readlink("/proc/self/exe", buffer, buffer_size);
	if (res == -1)
	{
		return false;
	}
	buffer[res] = 0;

	binary = buffer;
	return true;
}

pid_t get_current_process_id()
{
	return getpid();
}

#else

void sleep_ms(int ms)
{
	Sleep(ms);
}

bool get_binary_name(string& binary)
{
	bool bResult = false;

	do
	{
		binary.clear();
		binary.resize( MAX_PATH, 0 );
		HMODULE hModule = GetModuleHandle(NULL);
		if (!hModule)
		{
			break;
		}

		if (!GetModuleFileNameA(hModule, (char *)binary.c_str(), MAX_PATH))
		{
			break;
		}

		bResult = true;
	}
	while(false);

	return bResult;
}

pid_t get_current_process_id()
{
	return GetCurrentProcessId();
}

#endif
}
