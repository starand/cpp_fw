#include "stdafx.h"
#include "worker.h"
#include "logerror.h"

//////////////////////////////////////////////////
// CWorker implementation

CWorker::CWorker( CSocket* pSocket ) : m_pSocket(pSocket)
{
	_ASSERT( m_pSocket );
}

CWorker::~CWorker()
{
	delete m_pSocket;
}

bool CWorker::Execute()
{
	if( m_pSocket )
	{
		OnExecute();
	}
	else
	{
		_ASSERT( m_pSocket );
		LOG_ERROR( "m_pSocket is NULL" );
	}

	delete this;
	return true;
}
