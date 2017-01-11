#include "StdAfx.h"
#include "mutex.h"

CMutex::CMutex() : m_hMutex(NULL), m_bLocked(false)
{
	m_hMutex = CreateMutex(NULL, false, NULL);
}

CMutex::~CMutex()
{
	m_dwWaitResult = WaitForSingleObject(m_hMutex, INFINITE);
	if( m_hMutex )
	{
		m_bLocked = false;

		CloseHandle(m_hMutex);
		m_hMutex = NULL;
	}
}

void CMutex::Lock()
{
	if( m_hMutex )
	{
		m_dwWaitResult = WaitForSingleObject(m_hMutex, INFINITE);
		m_bLocked = true;
	}
}

void CMutex::Unlock()
{
	if( m_hMutex )
	{
		ReleaseMutex( m_hMutex );
		m_bLocked = false;
	}
}

