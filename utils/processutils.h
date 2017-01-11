#ifndef __H_PROCESSUTILS__
#define __H_PROCESSUTILS__

#include "types.h"

namespace ProcUtils
{
	bool GetProcessIDList( uint_v& vuiProcessList );
	bool GetModuleList( uint nProcessID, vector<HMODULE>& vuiModuleList );

	const char* GetModuleName( HMODULE hModule );
	const char* GetModuleName( uint nProcessID, HMODULE hModule );
};

#endif // __H_PROCESSUTILS__
