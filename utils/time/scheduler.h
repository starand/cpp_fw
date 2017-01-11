#ifndef __H_SCHEDULER__
#define __H_SCHEDULER__

#include "thread.h"
#include "datetime.h"
#include "priorityqueue.h"
#include "criticalsection.h"
#include "tcpserver_mt.h"

#define DEFAULT_SLEEPTIMEOUT 1000

class ITask;
class CTaskReceiver;

class CScheduler : public CThread
{
public:
	CScheduler( ushort nPort, size_t nTimeout = DEFAULT_SLEEPTIMEOUT );

	bool Start();
	bool Execute();

	bool AddTask( const CDateTime& dtTime, const ITask* tTask, const CDateTime& dtRepeatTime );

private:
	struct CListItem
	{
		const CDateTime m_dtTime;
		const ITask* m_pTask;
		const CDateTime m_dtRepeatTime;

		CListItem( const CDateTime dtTime, const ITask* tTask, const CDateTime dtRepeatTime );
		bool operator<( const CListItem& liRight ) const;
	};

	CPriorityQueue<CListItem> m_liItems;
	size_t	m_nTimeout;

	CCriticalSection	m_ccMutex;

	CTaskReceiver*		m_ptrTaskReceiver;
	ushort				m_nPort;
	
};

#endif // __H_SCHEDULER__
