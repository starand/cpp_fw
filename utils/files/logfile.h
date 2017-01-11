#ifndef __LOGFILE_H_INCLUDED
#define __LOGFILE_H_INCLUDED

#include "loginterface.h"


extern CLogInterface* g_pLog;

class CCriticalSection;

//////////////////////////////////////////////////
// CLogFile declaration

class CLogFile : public CLogInterface
{
public:
	CLogFile( bool bFlush = true );
	~CLogFile();

	bool StartLogging( const char* szFileName );
	bool StopLogging();

	bool AddMessage( const char* szMessage, const char* szFunctionName, const char *szFileName, long uLine );

private:
	CCriticalSection &GetMutex() const { return const_cast<CCriticalSection &>(*m_pcsMutex); }

private:
	string	m_sLogFileName;
	FILE*	m_fpFile;
	bool	m_bFlush;

	CCriticalSection	*m_pcsMutex;
};

////////////////////////////////////////////////////////////////////////
// ************** To Start Logging Use This Function **************** //
////////////////////////////////////////////////////////////////////////

bool StartLogger(const string& sLogFileName);

#endif // __LOGFILE_H_INCLUDED
