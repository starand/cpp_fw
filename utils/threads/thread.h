#ifndef __H_THREADNEW__
#define __H_THREADNEW__

#include "taskinterface.h"
#include <Windows.h>

extern const char szStartThreadError[];

class CThread
{
public:
	CThread();
	virtual ~CThread();

	virtual bool Start(bool bSuspended = false);
	virtual bool Stop();
	virtual bool Suspend();
	virtual bool Resume();

	HANDLE GetHandle() const;
	DWORD GetID();

	friend DWORD WINAPI __THREAD_PROC__( void* pvParams );

	void wait(DWORD ms = INFINITE);

protected:
	virtual bool Execute() = 0;

private:
	DWORD	m_nThreadId;
	HANDLE	m_hThread;
};

typedef CThread thread_t;

#endif // __H_THREADNEW__
