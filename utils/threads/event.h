#ifndef __H_EVENT__
#define __H_EVENT__

#include "autohandle.h"

#define EV_FREE		true
#define EV_BLOCKED	false
#define EV_MANUALRESET true

class CEvent : public CAutoHandle
{
public:
	CEvent( const char* szEventName = NULL );
	CEvent( bool bInitialState, bool bManualReset = false, const char* szEventName = NULL );

	bool Signal();
	bool ResetEvent();

	DWORD WaitFor( DWORD dwMilliseconds = INFINITE );
	bool IsFree();
};

#endif // __H_EVENT__
