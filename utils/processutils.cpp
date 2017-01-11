#include "stdafx.h"
#include "processutils.h"

#include <Psapi.h>
#pragma comment( lib, "Psapi.lib" )

namespace ProcUtils {


bool GetProcessIDList( uint_v& vuiProcessList )
{
	bool bResult = false;

	do
	{
		vuiProcessList.clear();
		DWORD aProcesses[ 1024 ];
		DWORD cbNeeded;
		if( !EnumProcesses(aProcesses, sizeof(aProcesses), &cbNeeded) ) break;

		size_t nPorcessCount = cbNeeded >> 2;
		for( size_t idx = 0; idx < nPorcessCount; ++idx ) vuiProcessList.push_back( aProcesses[idx] );
		
		bResult = true;
	}
	while( false );

	return bResult;
}

bool GetModuleList( uint nProcessID, vector<HMODULE>& vuiModuleList )
{
	bool bResult = false;

	do
	{
		vuiModuleList.clear();
		HANDLE hProcess = OpenProcess( PROCESS_QUERY_INFORMATION | PROCESS_VM_READ, FALSE, nProcessID );
		if( !hProcess ) break;

		DWORD cbNeeded;
		HMODULE hModules[ 1024 ];
		if( !EnumProcessModules(hProcess, hModules, sizeof(hModules), &cbNeeded ) ) break;

		size_t nModsCount = cbNeeded >> 2;
		for( size_t idx = 0; idx < nModsCount; ++idx ) vuiModuleList.push_back( hModules[idx] );
	}
	while( false );

	return bResult;
}

const char* GetModuleName( HMODULE hModule )
{
	static char szModuleName[ 2048 ];
	if( !GetModuleFileNameA(hModule, szModuleName, 2048) ) return NULL;
	return (const char*)szModuleName;
}

const char* GetModuleName( uint nProcessID, HMODULE hModule )
{
	static char szModuleName[ 2048 ];
	HANDLE hProcess = OpenProcess( PROCESS_QUERY_INFORMATION | PROCESS_VM_READ, FALSE, nProcessID );
	if( !hProcess ) return NULL;

	if( !GetModuleFileNameExA( hProcess, hModule, szModuleName, 2048) ) return NULL;
	return (const char*)szModuleName;
}

}; // end of the ProcUtils namespace
