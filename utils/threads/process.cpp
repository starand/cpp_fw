#include "stdafx.h"
#include "process.h"

//////////////////////////////////////////////////
// CProcess implementation

CProcess::CProcess() 
{
	_Init();
}

CProcess::CProcess( const string& sCmdLine ) : m_sCmdLine(sCmdLine)
{
	_Init();
}

CProcess::CProcess( const string& sAppName, const string& sCmdLine ) 
	: m_sAppName(sAppName), m_sCmdLine(sCmdLine)
{
	_Init();
}

CProcess::~CProcess()
{
	CloseHandles();
}

void CProcess::_Init()
{
	ZeroMemory( &m_siStartupInfo, sizeof(m_siStartupInfo) );
	m_siStartupInfo.cb = sizeof(m_siStartupInfo);

	ZeroMemory( &m_piProcessInformation, sizeof(m_piProcessInformation) );


}

bool CProcess::Start()
{
	_ASSERT( !(m_sAppName.empty() && m_sCmdLine.empty()) );

	return CreateProcessA( m_sAppName.empty() ? NULL : m_sAppName.c_str(),
		m_sCmdLine.empty() ? NULL : (char*)m_sCmdLine.c_str(), 
		NULL, NULL, false, 0, NULL, NULL, &m_siStartupInfo, &m_piProcessInformation ) != 0;
}

bool CProcess::Start( const string& sCmdLine )
{
	_Init();
	m_sCmdLine.assign( sCmdLine );
	return Start();	
}

bool CProcess::Start( const string& sAppName, const string& sCmdLine )
{
	_Init();
	m_sAppName.assign( sAppName );
	m_sCmdLine.assign( sCmdLine );
	return Start();	
}

bool CProcess::Wait( uint uiTimeout /*= INFINITE*/ )
{
	return ( WAIT_OBJECT_0 == WaitForSingleObject( GetHandle(), uiTimeout ) );
}

uint CProcess::GetExitCode() const
{
	DWORD uiExitCode = 0;
	GetExitCodeProcess( GetHandle(), &uiExitCode );
	return (uint)uiExitCode;
}

uint CProcess::GetThreadExitCode() const
{
	DWORD uiExitCode = 0;
	GetExitCodeThread ( GetThreadHandle(), &uiExitCode );
	return (uint)uiExitCode;
}

bool CProcess::Terminate( uint uiExitCode /*= 0*/ )
{
	return TerminateProcess( GetHandle(), uiExitCode ) != 0;
}

void CProcess::CloseHandles()
{
	if( m_piProcessInformation.hProcess ) {
		CloseHandle( m_piProcessInformation.hProcess );
		m_piProcessInformation.hProcess = 0;
	}

	if( m_piProcessInformation.hThread ) {
		CloseHandle( m_piProcessInformation.hThread );
		m_piProcessInformation.hThread = 0;
	}
}
