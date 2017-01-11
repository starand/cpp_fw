#include "StdAfx.h"
#include "System.h"

#include <Windows.h>
#include <WinSock2.h>
#include <PowrProf.h>

#pragma comment(lib, "ws2_32.lib")
#pragma comment(lib, "PowrProf.lib")

//////////////////////////////////////////////////
// CSystem implementation

namespace System
{

uint SetStayOnMode(bool bOnMode)
{
	return bOnMode
		? SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_AWAYMODE_REQUIRED)
		: SetThreadExecutionState(ES_CONTINUOUS);
}

bool ConsoleDetach() 
{ 
	bool bResult = FreeConsole() != 0; 
	return bResult;
}


bool GetCompName(string& sOutCompName)
{
	char szBuffer[MAX_COMPUTERNAME_LENGTH + 1];
	DWORD dwBufferLength = MAX_COMPUTERNAME_LENGTH;

	bool bResult = false;
	if (GetComputerNameA(szBuffer, &dwBufferLength) != 0)
	{
		sOutCompName = szBuffer;
		bResult = true;
	}

	return bResult;
}

bool GetHostName(string& sOutHostName)
{
	START_FUNCTION_BOOL();

	WSAData wsaData;
	if (WSAStartup(MAKEWORD(1, 1), &wsaData) != 0) {
		break;
	}

	const sizeint siBufferLen = 80;
	char szBuffer[siBufferLen];

	if (gethostname(szBuffer, sizeof(szBuffer)) == SOCKET_ERROR) {
		break;
	}

	sOutHostName = szBuffer;

	END_FUNCTION_BOOL();
}

bool GetFirstIpAddr(string& sOutIpAddr)
{
	START_FUNCTION_BOOL();

	string sHostName;
	if (!GetHostName(sHostName)) {
		break;
	}

	struct hostent *pheHostEntry = gethostbyname(sHostName.c_str());
	if (!pheHostEntry) {
		break;
	}

	if (pheHostEntry->h_addr_list[0] == 0) {
		break;
	}

	struct in_addr iaAddress;
	memcpy(&iaAddress, pheHostEntry->h_addr_list[0], sizeof(struct in_addr));
	sOutIpAddr = inet_ntoa(iaAddress);

	END_FUNCTION_BOOL();
}

sizeint GetIpAddrList(string_v& vsOutIpAddrList)
{
	START_FUNCTION();
	string_v vsIpAddrList;

	string sHostName;
	if (!GetHostName(sHostName)) {
		break;
	}

	struct hostent *pheHostEntry = gethostbyname(sHostName.c_str());
	if (!pheHostEntry) {
		break;
	}

	struct in_addr addr;
	for (int idx = 0; pheHostEntry->h_addr_list[idx] != 0; ++idx) 
	{
		memcpy(&addr, pheHostEntry->h_addr_list[idx], sizeof(struct in_addr));
		vsIpAddrList.push_back(inet_ntoa(addr));
	}

	vsOutIpAddrList.swap(vsIpAddrList);
	END_FUNCTION_RET(vsOutIpAddrList.size());
}


bool LockWorkStation()
{
	bool bResult = ::LockWorkStation() != 0;
	return bResult;
}

bool Hibernate()
{
	return SetSuspendState(true, true, false) != 0;
}

bool Sleep()
{
	return SetSuspendState(false, true, false) != 0;
}

bool PowerOff()
{
	ASSERT_NOT_IMPLEMENTED();
	return true;
}


namespace Windows
{

bool TerminateWindow(HWND hWnd)
{
	bool    bResult = false;

	DWORD   dwProcessId = 0;
	HANDLE  hProcess = NULL;

	GetWindowThreadProcessId(hWnd, &dwProcessId);
	hProcess = OpenProcess(PROCESS_TERMINATE, FALSE, dwProcessId);

	if (hProcess != NULL)
	{
		if(TerminateProcess(hProcess, 0))
		{
			bResult = true;
		}
		CloseHandle(hProcess);
	}

	return bResult;
}

bool CloseForegroundWindow()
{
	HWND hWnd = GetForegroundWindow();
	
	bool bResult = TerminateWindow(hWnd);
	return bResult;
}

}; // namespace Windows

}; // namespace CSystem
