#include "stdafx.h"
#include "tcpserver.h"
#include "logerror.h"
#include "consts.h"

//////////////////////////////////////////////////
// CTCPWorker implementation

CTCPWorker::CTCPWorker( CSocket* pSocket, CTCPServer* pTCPServer ) : 
	m_pSocket(pSocket), m_pTCPServer(pTCPServer) 
{

}

CTCPWorker::~CTCPWorker() { delete m_pSocket; }

bool CTCPWorker::Execute()
{
	do
	{
		if( !m_pSocket ) {
			_ASSERT_BREAK( m_pSocket );
		}

		if( !m_pTCPServer ) {
			_ASSERT_BREAK( m_pTCPServer );
		}

		const_cast<CTCPServer*>(m_pTCPServer)->OnClientRequest( const_cast<CSocket*>(m_pSocket) );
	}
	while( false );

	delete this;
	return true;
}

//////////////////////////////////////////////////
// CReceiver implementation

void CTCPServer::OnClientAccepted( CSocket* psClient )
{
	START_FUNCTION();
	_ASSERT_BREAK( psClient );

	CTCPWorker* pWorker = new (nothrow) CTCPWorker( psClient, this );
	if( !pWorker ) {
		LOG_ERROR( szUnableToAllocateMemory );
		_ASSERT_BREAK( pWorker );
	}

	if( !pWorker->Start() ) {
		LOG_ERROR_BREAK( szUnableToStartThread );
	}

	END_FUNCTION();
}