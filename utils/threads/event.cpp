#include "StdAfx.h"
#include "event.h"

////////////////////////////////////////////////////
// CEvent implementation

CEvent::CEvent( const char* szEventName /*= NULL*/ )
	: CAutoHandle(CreateEventA(NULL, FALSE, FALSE, szEventName))
{
}

CEvent::CEvent( bool bInitialState, bool bManualReset /*= false*/, const char* szEventName /*= NULL*/ )
	: CAutoHandle(CreateEventA(NULL, bManualReset, bInitialState, szEventName))
{
}

bool CEvent::Signal()
{
	return ( ::SetEvent( m_Handle ) != 0 );
}

bool CEvent::ResetEvent()
{
	return ( ::ResetEvent( m_Handle ) != 0 );
}

DWORD CEvent::WaitFor( DWORD dwMilliseconds /*= INFINITE*/ )
{
	return ::WaitForSingleObject( m_Handle, dwMilliseconds );
}

bool CEvent::IsFree()
{
	bool bResult = false;
	if( WAIT_OBJECT_0 == WaitForSingleObject(m_Handle, 0)  )
	{
		SetEvent(m_Handle);
		bResult = true;
	}

	return bResult;
}
