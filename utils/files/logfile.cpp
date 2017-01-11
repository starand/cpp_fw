#include "stdafx.h"
#include "logfile.h"
#include "macroes.h"
#include "autolock.h"
#include "datetime.h"
#include "consts.h"
#include "strutils.h"
#include "CriticalSection.h"
#include "fileutils.h"

using namespace StrUtils;

CLogInterface* g_pLog = NULL;

//////////////////////////////////////////////////
// CLogFile implementation

CLogFile::CLogFile( bool bFlush /*= true*/ ):
	m_fpFile(), 
	m_bFlush(bFlush),
	m_pcsMutex(NULL)
{
	m_pcsMutex = new (nothrow) CCriticalSection();
	ASSERTE(m_pcsMutex);
}

CLogFile::~CLogFile()
{
	StopLogging();
	delete m_pcsMutex;
}

bool CLogFile::StartLogging( const char* szFileName )
{
	BOOL_FUNCTION_START();
	_ASSERT_BREAK( szFileName && szFileName[0] );

	m_sLogFileName.assign( szFileName );

#pragma warning( disable : 4996 )
	m_fpFile = fopen( m_sLogFileName.c_str(), "w" );
#pragma warning( default : 4996 )

	if( !m_fpFile ) break;
	BOOL_FUNCTION_END();
}

bool CLogFile::StopLogging()
{
	return ( m_fpFile && 0 == fclose(m_fpFile) && !(m_fpFile = NULL) );
}


bool CLogFile::AddMessage(const char* szMessage, const char* szFunctionName, const char *szFileName, long uLine)
{
	string sMessage;
	
	string sTime = CDateTime().GetTimeStr();
	sTime.append(".").append(CDateTime().GetTimeStr(EDTF_SECMS));

	FormatString(sMessage, "%s [%s:%u]: %s\n", sTime.c_str(), szFunctionName, uLine, szMessage );

	CAutoLock<> lockObj(GetMutex());
	bool bResult = ( m_fpFile && szMessage && EOF != fputs(sMessage.c_str(), m_fpFile) );
	if( bResult && m_bFlush ) fflush( m_fpFile );
	return bResult;
}

//////////////////////////////////////////////////////////////////////////

bool StartLogger( const string& sLogFileName )
{
	START_FUNCTION_BOOL();
	ASSERTE(!sLogFileName.empty());

	if( g_pLog ) {
		LOG_ERROR_BREAK( "Logger is already initialized" );
	}

	g_pLog = new (nothrow) CLogFile();
	if( !g_pLog ) {
		LOG_ERROR_BREAK( "Unable to allocate memory" );
	}

	if( !g_pLog->StartLogging(sLogFileName.c_str()) ) 
	{
		delete g_pLog;
		LOG_ERROR2( "Unable to open log file %s", sLogFileName.c_str() );
	}

	END_FUNCTION_BOOL();
}
