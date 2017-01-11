#include "StdAfx.h"
#include "thread.h"
#include "logerror.h"

static const char szStartThreadError[] = "Unable to start thread";

//////////////////////////////////////////////////
// CThread implementation

DWORD WINAPI __THREAD_PROC__( void* pvParams )
{
	return ( pvParams && ((CThread*)pvParams)->Execute() ? 1 : 0 );
}

CThread::CThread() : m_nThreadId(), m_hThread()
{
}

CThread::~CThread()
{
}

bool CThread::Start(bool bSuspended/* = false*/)
{
	DWORD dwCreationFlags = bSuspended ? CREATE_SUSPENDED : 0;

	m_hThread = CreateThread( NULL, 0, __THREAD_PROC__, this, dwCreationFlags, &m_nThreadId );
	if( !m_hThread )
	{
		LOG_ERROR( szStartThreadError );
	}

	return ( NULL != m_hThread );
}

bool CThread::Stop()
{
	return ( TerminateThread( m_hThread, 0 ) == TRUE );
}

bool CThread::Suspend()
{
	return ( -1 != SuspendThread(m_hThread) );
}

bool CThread::Resume()
{
	return ( -1 != ResumeThread(m_hThread) );
}

HANDLE CThread::GetHandle() const
{
	return m_hThread;
}

DWORD CThread::GetID()
{
	return m_nThreadId;
}


void CThread::wait(DWORD ms /*= INFINITE*/)
{
	::WaitForSingleObject(m_hThread, ms);
}
