#include "stdafx.h"
#include "scheduler.h"
#include "datetime.h"
#include "logerror.h"
#include "autolock.h"
#include "taskreceiver.h"
#include "consts.h"

/////////////////////////////////////////////////
// CScheduler implementation

CScheduler::CScheduler( ushort nPort, size_t nTimeout /*= DEFAULT_SLEEPTIMEOUT*/ ) : m_nTimeout(nTimeout), m_nPort(nPort) 
{
	m_ptrTaskReceiver = new(nothrow) CTaskReceiver();
	_ASSERT( m_ptrTaskReceiver );
}

bool CScheduler::Start()
{
	BOOL_FUNCTION_START(  );

	if( !CThread::Start() ) {
		LOG_ERROR_BREAK( szUnableToStartThread );
	}

	if( !m_ptrTaskReceiver ) {
		LOG_ERROR_BREAK( szUnableToAllocateMemory );
	}

	if( !m_ptrTaskReceiver->Start(m_nPort) ) break;

	BOOL_FUNCTION_END();
}

bool CScheduler::Execute()
{
	while( true )
	{
		if( m_liItems.empty() )
		{
			Sleep( m_nTimeout );
			continue;	
		}

		m_ccMutex.Lock();
		int64 nTimeDiff = m_liItems.begin()->m_dtTime.ToMillisecs() - CDateTime().ToMillisecs();
		if( nTimeDiff > 0 )
		{
			m_ccMutex.Unlock();
			Sleep( (uint)min(nTimeDiff, m_nTimeout) );
			continue;				
		}

		CListItem liItem = m_liItems.pop();
		m_ccMutex.Unlock();

		LOG_MSG( "Execute task");
		const_cast<ITask*>(liItem.m_pTask)->Execute();
				
		string sDescription;
		liItem.m_pTask->GetDescription( sDescription );
		if( !sDescription.empty() ) {
			LOG_TRACE3( "    %s at %s", sDescription.c_str(), liItem.m_dtTime.GetTimeStr()  );
		}

		if( !liItem.m_dtRepeatTime.GetIsNULL() )
		{
			CDateTime dtRepeatTime = liItem.m_dtTime + liItem.m_dtRepeatTime;
			LOG_TRACE2(  "  *** REPEAT at %s", dtRepeatTime.GetTimeStr() );
			CAutoLock<> lockObj( m_ccMutex );
			m_liItems.push( CListItem(dtRepeatTime, liItem.m_pTask, liItem.m_dtRepeatTime) );
		}
		else
		{
			delete liItem.m_pTask;
		}
	}

	return true;
}

bool CScheduler::AddTask( const CDateTime& dtTime, const ITask* tTask, const CDateTime& dtRepeatTime )
{
	bool bResult = false;

	do
	{
		CDateTime dtNow;
		string sCurrentTime = CDateTime().GetTimeStr();
		if( dtTime < dtNow ) LOG_ERROR3_BREAK( "Time in past : %s, now is %s", dtTime.GetTimeStr(), sCurrentTime.c_str() );
		if( !tTask ) LOG_ERROR_BREAK( "Invalid task pointer" );

		CAutoLock<CCriticalSection> lock( m_ccMutex );
		m_liItems.push( CListItem(dtTime, tTask, dtRepeatTime) );

		string sTaskDescription;
		tTask->GetDescription( sTaskDescription );
		LOG_MSG3( "Added new task [ %s ] in %s", sTaskDescription.c_str(), dtTime.GetTimeStr() );

		bResult = true;
	}
	while( false );

	return bResult;
}

/////////////////////////////////////////////////
// CListItem implementation

CScheduler::CListItem::CListItem( const CDateTime dtTime, const ITask* tTask, const CDateTime dtRepeatTime ) 
	: m_dtTime(dtTime), m_pTask(tTask), m_dtRepeatTime(dtRepeatTime)
{
}

bool CScheduler::CListItem::operator<( const CListItem& liRight ) const
{
	return m_dtTime < liRight.m_dtTime;
}
