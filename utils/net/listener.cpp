#include "stdafx.h"
#include "listener.h"
#include "logerror.h"
#include "consts.h"

static const char szPointerIsNULL[] = "Pointer is NULL";

//////////////////////////////////////////////////
// CListener implementation

CListener::CListener( bool bDeleteSocket /*= false*/ ) : m_bDeleteSocket(bDeleteSocket), m_pSocket()
{
	m_pSocket = new (nothrow) CSocket();
}

CListener::~CListener()
{
	delete m_pSocket;
}

bool CListener::Execute()
{
	BOOL_FUNCTION_START();

	if( !m_pSocket ) {
		_ASSERT( m_pSocket  );
		LOG_ERROR2_BREAK( "m_pSocket %s", szPointerIsNULL );
	}

	while( true )
	{
		CSocket* psClient = NULL;
		while( psClient = m_pSocket->Accept() )
		{
			OnClientAccepted( psClient );
			if( m_bDeleteSocket ) FREE_POINTER( psClient );
		}

		_ASSERT( psClient );
		LOG_ERROR( g_szUnableToAcceptClient );
	}

	BOOL_FUNCTION_END();
}

bool CListener::Start( ushort nPort, const string& sBindAddress /*= DEF_BIND_ADDRESS*/ )
{
	BOOL_FUNCTION_START();
	_ASSERT_BREAK( m_pSocket );

	if (!m_pSocket->Listen(nPort, sBindAddress.c_str()))
	{
		LOG_ERROR3_BREAK(g_szUnableToStartListening, sBindAddress.c_str(), nPort);
	}

	if (!CThread::Start())
	{
		LOG_ERROR_BREAK(szStartThreadError);
	}

	BOOL_FUNCTION_END();
}
