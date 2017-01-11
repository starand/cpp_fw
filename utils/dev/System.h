#ifndef __SYSTEM_H_INCLUDED
#define __SYSTEM_H_INCLUDED

//////////////////////////////////////////////////
// CSystem declaration

namespace System
{
	uint SetStayOnMode(bool bOnMode);
	bool ConsoleDetach();

	bool GetCompName(string& sOutCompName);
	bool GetHostName(string& sOutHostName);
	
	bool GetFirstIpAddr(string& sOutIpAddr);
	sizeint GetIpAddrList(string_v& vsOutIpAddrList);

	bool LockWorkStation();

	bool Hibernate();
	bool Sleep();
	bool PowerOff();

	namespace Windows
	{
		bool TerminateWindow(HWND hWnd);
		bool CloseForegroundWindow();
	}
};

#endif // __SYSTEM_H_INCLUDED
