#ifndef __H_SEMAPHORE__
#define __H_SEMAPHORE__

#include "autohandle.h"

class CSemaphore : public CAutoHandle
{
public:
	CSemaphore( DWORD dwMaxCount, DWORD dwInitCount = 0, const char* szEventName = NULL )
		: CAutoHandle(CreateSemaphoreA(NULL, dwInitCount, dwMaxCount, szEventName)) { }

	bool Release( DWORD dwCount = 1 )
	{
		return ( ::ReleaseSemaphore( m_Handle, dwCount, NULL ) != 0 );
	}

	DWORD WaitFor( DWORD dwMilliseconds = INFINITE )
	{
		return ::WaitForSingleObject( m_Handle, dwMilliseconds );
	}

	bool Wait( DWORD dwMilliseconds = INFINITE )
	{
		return ( WAIT_OBJECT_0 ==  ::WaitForSingleObject(m_Handle, dwMilliseconds) );
	}

};

#endif // __H_SEMAPHORE__
