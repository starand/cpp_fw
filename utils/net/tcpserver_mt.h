#ifndef __H_TCPSERVER_MT__
#define __H_TCPSERVER_MT__

#include "listener.h"
#include "worker.h"
#include "logerror.h"
#include "consts.h"


//////////////////////////////////////////////////
// IExecutor definition

class IClientHandler
{
protected:
	virtual void OnExecute( CSocket* pSocket ) = 0;
};

//////////////////////////////////////////////////
// CExecutor definition

template<class TExecuteStrategy>
class CExecutor : public CWorker, public TExecuteStrategy
{
public:
	CExecutor( CSocket* pSocket ) : CWorker(pSocket) { }

protected:
	void OnExecute()
	{
		if( m_pSocket ) {
			TExecuteStrategy::OnExecute( const_cast<CSocket*>(m_pSocket) );
		}
	}
};


//////////////////////////////////////////////////
// CReceiver definition

template<typename TExecuteStrategy> 
class CTCPServerMT : public CListener
{
protected:
	typedef CExecutor<TExecuteStrategy> TExecutor;

	void OnClientAccepted( CSocket* psClient )
	{
		START_FUNCTION();
		_ASSERT_BREAK( psClient );

		TExecutor* pNewExecutor = new (nothrow) TExecutor( psClient );
		if( !pNewExecutor ) {
			LOG_ERROR( "Unable to allocate memory" );
			_ASSERT_BREAK( pNewExecutor );
		}

		if( !pNewExecutor->Start() ) {
			LOG_ERROR_BREAK( "Unable to start worker" );
		}

		END_FUNCTION();
	}
};

#endif // __H_TCPSERVER_MT__
