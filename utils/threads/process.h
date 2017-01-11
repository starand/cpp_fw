#ifndef __H_PROCESS__
#define __H_PROCESS__

#include <Windows.h>

class CProcess
{
public:
	CProcess();
	CProcess( const string& sCmdLine );
	CProcess( const string& sAppName, const string& sCmdLine );
	~CProcess();

	bool Start();
	bool Start( const string& sCmdLine );
	bool Start( const string& sAppName, const string& sCmdLine );
	bool Terminate( uint uiExitCode = 0 );
	void CloseHandles();

	uint GetErrorCode() const { return GetLastError(); }
	uint GetExitCode() const;
	uint GetThreadExitCode() const;

	bool Wait( uint uiTimeout = INFINITE );

	const HANDLE GetHandle() const { return m_piProcessInformation.hProcess; }
	const uint GetID() const { return m_piProcessInformation.dwProcessId; } 

	const HANDLE GetThreadHandle() const { return m_piProcessInformation.hThread; }
	const uint GetThreadID() const { return m_piProcessInformation.dwThreadId; }

protected:
	void _Init();

private:
	string m_sAppName;
	string m_sCmdLine;

	STARTUPINFOA		m_siStartupInfo;
    PROCESS_INFORMATION	m_piProcessInformation;
};

#endif // __H_PROCESS__
