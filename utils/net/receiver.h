#ifndef __H_RECEIVER__
#define __H_RECEIVER__

#include "socket.h"
#include "thread.h"
#include "logerror.h"
#include "macroes.h"

///////////////////////////////////////////////////////
// CReceiver emaple
//
//class CRecvTask : public CThread
//{
//public:
//	CRecvTask( CSocket* pSock = NULL ) : m_pSocket(pSock) {}
//	virtual ~CRecvTask() { FREE_POINTER( m_pSocket ); }
//
//	bool Execute() {
//		// To do something
//		delete this;
//	}
//
//protected:
//	CSocket*	m_pSocket;
//};


template<typename T>
class CReceiver : 
	protected CSocket, 
	protected CThread
{
public:
	CReceiver( ushort nPort, const char* szAddress = "0.0.0.0", int nMaxClient = MAXCONN )
		: m_nPort(nPort), m_sAddress(szAddress), m_nMaxClient(nMaxClient) 
	{ }

	bool Start() 
	{
		bool bResult = false;
		do
		{
			if( !Listen(m_nPort, m_sAddress.c_str(), m_nMaxClient) ) 
				LOG_ERROR3_BREAK( "Unable to listen %s:%u", m_sAddress.c_str(), m_nPort );

			if( !CThread::Start() ) LOG_ERROR_BREAK( "Unable to start receiver thread" );
			bResult = true;
		}
		while( false );
		return bResult;
	}

	bool Stop() 
	{
		bool bResult = false;
	
		do
		{
			if( !CThread::Stop() ) LOG_ERROR( "Unable to stop receiver thread" );
			if( !CSocket::Close() ) LOG_ERROR( "Unable to close receiver socket" );
			bResult = true;
		}
		while( false );

		return bResult;
	}

protected:
	bool Execute()
	{
		while( CSocket* pClientSock = Accept() )
		{
			LOG_MSG3( "Client %s:%hu accepted", pClientSock->LocalAddr(), pClientSock->LocalPort() );
		
			T* pThread = new T( pClientSock );
			pThread->Start();
		}
		return true;
	}

private:
	ushort	m_nPort;
	string	m_sAddress;
	int		m_nMaxClient;
};

#endif // __H_RECEIVER__
