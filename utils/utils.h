#ifndef __UTILS_H_INCLUDED
#define __UTILS_H_INCLUDED

#include <string>

#ifdef WINDOWS
#	ifndef _PID_T_
		typedef DWORD pid_t;
#	endif
#endif

namespace utils
{
	void sleep_ms(int ms);

	bool get_binary_name(std::string& binary);
	pid_t get_current_process_id();
};

#endif // __UTILS_H_INCLUDED
