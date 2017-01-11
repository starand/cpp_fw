#ifndef __H_TASKQUEUE__
#define __H_TASKQUEUE__

#include <queue>
#include "mutex.h"
#include "event.h"

template<typename CTask>
class CTaskQueue : protected queue<CTask>
{
public:
	void AddTask( const CTask& tTask )
	{
		CAutoLock<CMutex> alLock( m_Mutex );
		push( tTask );
	}

	bool GetNextTask( CTask& tTask )
	{
		bool bResult = false;

		if( !empty() )
		{
			CAutoLock<CMutex> alLock( m_Mutex );
			if( !empty() )
			{
				tTask = front();
				pop();
				bResult = true;
			}
		}
	
		return bResult;
	}

private:
	CMutex m_Mutex;
};

#endif // __H_TASKQUEUE__
