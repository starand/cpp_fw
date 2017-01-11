#ifndef __H_LOG_INTERFACE__
#define __H_LOG_INTERFACE__

class CLogInterface
{
public:
	virtual ~CLogInterface() { }

	virtual bool StartLogging( const char* szFileName ) = 0;
	virtual bool AddMessage( const char* szMessage, const char* szFunctionName, const char *szFileName, long uLine ) = 0;
};

#endif // __H_LOG_INTERFACE__
