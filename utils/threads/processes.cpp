#include "stdafx.h"
#include "processes.h"
#include "Shellapi.h"

#define SYSTEM_ERROR -1

//////////////////////////////////////////////////
// 

bool RunCommandLine( const char* szCommand )
{
	return ( SYSTEM_ERROR != system(szCommand) ) ;
}

bool StartProcess( const char* szCommand )
{
	static size_t nBufferLen = 260;
	static char szCommandBuffer[260] = { 0 };
	
	BOOL_FUNCTION_START();

	if( !szCommand || *szCommand == 0 ) break;
	sprintf_s( szCommandBuffer, 260, "start %s", szCommand );

	if( SYSTEM_ERROR == system(szCommandBuffer) ) break;

	BOOL_FUNCTION_END();
}

bool ExecuteCommand( const string& sCommandLine, bool bHide /*= false*/ )
{
	string sParams, sCommand = ParseCommandParams( sCommandLine, sParams );
	return (int)ShellExecuteA(NULL, "open", sCommand.c_str(), sParams.c_str(), NULL, bHide ? SW_HIDE : SW_SHOW) > 32;
}

//////////////////////////////////////////////////
// CProcessTask implementation

CProcessTask::CProcessTask( const char* szTaskCmd ) 
	: m_sTaskCommand(szTaskCmd) 
{ 
}

bool CProcessTask::Execute() 
{ 
	return StartProcess( m_sTaskCommand.c_str() ); 
}


void CProcessTask::GetDescription( string& sDescription ) const
{
	sDescription.assign( m_sTaskCommand );
}
