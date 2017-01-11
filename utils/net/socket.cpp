#include "StdAfx.h"
#include "socket.h"
#include "netutils.h"
#include "logerror.h"
#include "Asserts.h"

#pragma comment(lib, "ws2_32.lib")

#ifndef MAX_RETRIEVED_DATA_LEN
#	define MAX_RETRIEVED_DATA_LEN	0xFFFF
#endif

#ifndef SHOW_SOCKET_ERRORS
#	undef LOG_ERROR3
#	define LOG_ERROR3(x,p1,p2)
#	undef LOG_ERROR_BREAK
#	define LOG_ERROR_BREAK(x) break;
#	undef LOG_ERROR3_BREAK
#	define LOG_ERROR3_BREAK(x,p1,p2) break;
#	undef LOG_ERROR4_BREAK
#	define LOG_ERROR4_BREAK(x,p1,p2,p3) break;
#endif


const char szErrorFailedWithCodeFmt[] = "%s() failed with error %i";
const char szSocketNotReady[] = "Socket is not ready";
const char szMemoryAllocationError[] = "Memory Allocation Error";

const char szEmptyAddress[] = "";

WSADATA CSocket::m_wsaData;
bool CSocket::m_bInitialized = false;

bool CSocket::InitSockets() 
{
	do
	{
		m_bConnected = false;
		m_nRwTimeOut = DEFAULT_RW_TIMEOUT;
		m_nLingerTimeout = DEFAULT_LINGER_TIMEOUT;
		m_nWaitNextTimeOut = DEFAULT_WAITNEXT_TIMEOUT;
		m_nCheckAttempts = DEFAULT_CHECK_ATTEMPTS;

		memset( &m_saLocalAddr, 0, sizeof(m_saLocalAddr) );
		memset( &m_saRemoteAddr, 0, sizeof(m_saRemoteAddr) );

		m_RSALen = sizeof( m_saRemoteAddr );

		if( m_bInitialized ) break;
		if( WSAStartup(MAKEWORD(1, 1), &m_wsaData) ) {
			LOG_ERROR3_BREAK( szErrorFailedWithCodeFmt, "WSAStartup", WSAGetLastError() );
		}
	
		m_bInitialized = true;
	}
	while( false );

	return m_bInitialized;
}

CSocket::CSocket( bool bUseLength /* = false */ ) : m_bUseHeaderLength(bUseLength)
{
	InitSockets();
}

CSocket::~CSocket()
{
	Close();
}

bool CSocket::Connect( const string& sHostName, short nPort )
{
	bool bResult = false;

	do
	{
		if( sHostName.empty() ) LOG_ERROR_BREAK( "Host name is empty" );
		string sHost = GetIpByName( sHostName.c_str() );
		// if( sHost.empty() ) LOG_ERROR2_BREAK( "Unable to retrieve IP for %s", sHostName.c_str() );

		m_saLocalAddr.sin_family = AF_INET;
		m_saLocalAddr.sin_addr.s_addr = inet_addr( sHost.c_str() );
		m_saLocalAddr.sin_port = htons( nPort );
		
		if( INVALID_SOCKET == (m_sockMain = socket( AF_INET, SOCK_STREAM, 0 )) ) {
			LOG_ERROR3_BREAK( szErrorFailedWithCodeFmt, "socket", WSAGetLastError() );
		}

		if( SOCKET_ERROR == connect(m_sockMain, (SOCKADDR *)&m_saLocalAddr, sizeof(m_saLocalAddr)) ) {
			LOG_ERROR4_BREAK("Unable to connect to %s:%u (%u)", sHost.c_str(), nPort, WSAGetLastError());
		}

		bResult = m_bConnected = m_bClientMode = true;
	}
	while(false);

	return bResult;
}

bool CSocket::Listen( ushort nPort, long lAddr, int nMaxClient )
{
	bool bResult = false;
	
	do
	{
		m_saLocalAddr.sin_addr.s_addr = htonl( lAddr );
		m_saLocalAddr.sin_port = htons( nPort );
		m_saLocalAddr.sin_family = AF_INET;

		if( INVALID_SOCKET == (m_sockMain = socket(AF_INET, SOCK_STREAM, 0)) ) {
			LOG_ERROR3_BREAK( szErrorFailedWithCodeFmt, "socket", WSAGetLastError() );
		}

		int nOption = 1;
		// if server will fall and client will be connected socket can be use address again
		if( SOCKET_ERROR == setsockopt(m_sockMain, SOL_SOCKET, SO_REUSEADDR, (char *)&nOption, sizeof(nOption)) ) {
			LOG_ERROR3_BREAK( szErrorFailedWithCodeFmt, "setsockopt", WSAGetLastError() );
		}

		if( SOCKET_ERROR == bind(m_sockMain, (struct sockaddr *)&m_saLocalAddr, sizeof(m_saLocalAddr)) ) {
			LOG_ERROR3_BREAK( szErrorFailedWithCodeFmt, "bind", WSAGetLastError() );
		}

		if( SOCKET_ERROR == listen(m_sockMain, nMaxClient) ) {
			LOG_ERROR3_BREAK( szErrorFailedWithCodeFmt, "listen", WSAGetLastError() );
		}

		m_bClientMode = false;
		bResult = true;
	}
	while( false );

	return bResult;
}

bool CSocket::Listen( ushort nPort, const char* szAddress, int nMaxClient /*= MAXCONN*/ )
{
	UINT uAddress = NetUtils::Ip2Uint( szAddress );
	return Listen( nPort, uAddress, nMaxClient );
}

CSocket* CSocket::Accept()
{
	CSocket* pClientSock = NULL;

	do
	{
		if( INVALID_SOCKET == (m_sockClient = accept(m_sockMain, (struct sockaddr *) &m_saRemoteAddr, (int *)&m_RSALen)) ) 
		{
			int nLastErrorCode = WSAGetLastError();
			LOG_ERROR3_BREAK( szErrorFailedWithCodeFmt, "accept", nLastErrorCode);
		}

		if( !(pClientSock = new (nothrow) CSocket(m_bUseHeaderLength)) )
		{	
			closesocket( m_sockClient );
			m_sockClient = INVALID_SOCKET;
			LOG_ERROR_BREAK( szMemoryAllocationError );
		}

		pClientSock->m_sockMain = m_sockClient;
		pClientSock->m_saLocalAddr = m_saLocalAddr;
		pClientSock->m_saRemoteAddr = m_saRemoteAddr;
		pClientSock->m_nRwTimeOut = m_nRwTimeOut;
		pClientSock->m_nLingerTimeout = m_nLingerTimeout;
		pClientSock->m_bConnected = true;
	}
	while( false );

	return pClientSock;
}

bool CSocket::Close()
{
	bool bResult = false;

	if( INVALID_SOCKET != m_sockMain )
	{
		linger strLinger = { 1, m_nLingerTimeout };
		setsockopt( m_sockMain, SOL_SOCKET, SO_LINGER, (char*)&strLinger, sizeof(strLinger) );

		shutdown( m_sockMain, SD_BOTH );
		bResult = ( 0 == closesocket(m_sockMain) );
		m_sockMain = INVALID_SOCKET;
	}

	return bResult;
}

bool CSocket::Send( const char* pszData, long nLen )
{
	bool bResult = false;

	do
	{
		bool bLoopError = false;
		int nSendLen = 0, nRes = 0, nNetLen = htonl( nLen );

		m_nBytesWritten = 0;

		if( !nLen ) LOG_ERROR_BREAK( "Send data is empty" );
		if( !pszData ) LOG_ERROR_BREAK( "pszData is NULL" );

		m_tvTimeOut.tv_sec  = (long)m_nRwTimeOut;
		m_tvTimeOut.tv_usec = 0;

		FD_ZERO( &m_fdWriteSockets );
		FD_SET( m_sockMain, &m_fdWriteSockets );
		if( SOCKET_ERROR == (nRes = select(int(m_sockMain)+1, 0, &m_fdWriteSockets, 0, &m_tvTimeOut)) ) {
			LOG_ERROR3_BREAK( szErrorFailedWithCodeFmt, "select", WSAGetLastError() );
		}

		if( !nRes ) LOG_ERROR_BREAK( szSocketNotReady );
		if( !FD_ISSET(m_sockMain, &m_fdWriteSockets) ) LOG_ERROR_BREAK( "Timeout elapsed" );

		if( m_bUseHeaderLength )
		{	
			nSendLen = send( m_sockMain, (char*)&nNetLen, sizeof(nNetLen), 0 );
			if( SOCKET_ERROR == nSendLen || !nSendLen ) {
				LOG_ERROR3_BREAK( szErrorFailedWithCodeFmt, "send", WSAGetLastError() );
			}
		}

		while( nLen > 0 )
		{
			nSendLen = send( m_sockMain, pszData, nLen, 0 );
			if( SOCKET_ERROR == nSendLen || !nSendLen ) {
				LOG_ERROR3( szErrorFailedWithCodeFmt, "send", WSAGetLastError() );
				bLoopError = true;
				break;
			}
			nLen -= nSendLen;
			pszData += nSendLen;
			m_nBytesWritten += nSendLen;
		}

		if( bLoopError ) break;
		bResult = true;
	}
	while( false );

	return bResult;
}

bool CSocket::Recv( char* pszData, long nLen, bool bEmptyError /*= true*/ )
{
	bool bResult = false;
	static uint anTimeOuts[ DEFAULT_CHECK_ATTEMPTS ] = { 500, 2000, 5000 };

	do
	{
		bool bLoopError = false;
		long nExpectedLen = 0, nRecvLen = 0, nRes = 0, nNetLen = 0;

		m_nBytesRead = 0;

		if( !nLen ) LOG_ERROR_BREAK( "Receiving buffer size is 0" );
		if( !pszData ) LOG_ERROR_BREAK( "pszData is NULL" );

		size_t nAttempts = 0;
		do
		{
			m_tvTimeOut.tv_sec  = (long)anTimeOuts[nAttempts] / 1000;
			m_tvTimeOut.tv_usec = ( anTimeOuts[nAttempts] % 1000 ) * 1000;

			FD_ZERO(&m_fdReadSockets);
			FD_SET(m_sockMain, &m_fdReadSockets);
			if( SOCKET_ERROR == (nRes = select(int(m_sockMain)+1, &m_fdReadSockets, 0, 0, &m_tvTimeOut)) ) {
				LOG_ERROR3_BREAK( szErrorFailedWithCodeFmt, "select", WSAGetLastError() );
			}
		}
		while( !nRes && m_nCheckAttempts > nAttempts++ );
		if( !nRes ) break; // LOG_ERROR_BREAK( szSocketNotReady );

		if( !FD_ISSET(m_sockMain, &m_fdReadSockets) ) LOG_ERROR_BREAK( "Timeout elapsed" );

		nExpectedLen = nLen;
		if( m_bUseHeaderLength )
		{
			nRecvLen = recv( m_sockMain, (char*)&nNetLen, sizeof(nNetLen), 0 );
			if( SOCKET_ERROR == nRecvLen || !nRecvLen ) {
				LOG_ERROR3_BREAK( szErrorFailedWithCodeFmt, "recv", WSAGetLastError() );
			}

			nExpectedLen = ntohl( nNetLen );
			if( nExpectedLen > nLen ) {
				LOG_ERROR( "Too small receiving buffer" );
				nExpectedLen = nLen;
			}

			//cout << nExpectedLen << " (" << nNetLen << " : " << nRecvLen << ")";
		}

		while( nExpectedLen > 0 )
		{
			nRecvLen = recv( m_sockMain, pszData, nExpectedLen, 0 );
			if( !nRecvLen  ) 
			{
				if( !m_nBytesRead ) {
					// when it is not first read attempt we shouldn't show this error
					if( bEmptyError ) {
						LOG_ERROR( "Received 0 bytes" );
						bLoopError = true;
					}
				}
				break;
			}

			if( SOCKET_ERROR == nRecvLen || nRecvLen < 0 ) {
				bLoopError = true;
				LOG_ERROR3_BREAK( szErrorFailedWithCodeFmt, "recv", WSAGetLastError() );
			}

			nExpectedLen -= nRecvLen;
			pszData += nRecvLen;
			m_nBytesRead += nRecvLen;

			if (!nExpectedLen)
			{
				break;
			}
			
			sizeint siNextTimeout = m_bUseHeaderLength ? 5000 : m_nWaitNextTimeOut;
			if (!CanRead(siNextTimeout))
			{
				if (m_bUseHeaderLength)
				{
					LOG_ERROR2("Wait times out %u", siNextTimeout);
					bLoopError = true;
				}

				break;
			}
		}

		bResult = !bLoopError;
	}
	while( false );

	return bResult;
}

bool CSocket::CanRead( long nMillisec )
{
	bool bResult = false;

	do
	{
		m_tvTimeOut.tv_sec  = nMillisec / 1000;
		m_tvTimeOut.tv_usec = ( nMillisec % 1000 ) * 1000; // 1000 = 0.001 s

		FD_ZERO( &m_fdReadSockets );
		FD_SET( m_sockMain, &m_fdReadSockets );
		int nRes = select( int(m_sockMain)+1, &m_fdReadSockets, 0, 0, &m_tvTimeOut );

		if( SOCKET_ERROR == nRes ) LOG_ERROR3_BREAK( szErrorFailedWithCodeFmt, "select", WSAGetLastError() );
		if( !nRes ) break; // LOG_ERROR( szSocketNotReady );
		if( !FD_ISSET(m_sockMain, &m_fdReadSockets) ) LOG_ERROR_BREAK( "Timeout elapsed" );

		bResult = true;
	}
	while(false);

	return bResult;
}

bool CSocket::CanWrite( long nMillisec )
{
	int nRes = 0;

	do
	{
		m_tvTimeOut.tv_sec  = nMillisec / 1000;
		m_tvTimeOut.tv_usec = ( nMillisec % 1000 ) * 1000; // 1000 = 0.001 s

		FD_ZERO( &m_fdWriteSockets );
		FD_SET( m_sockMain, &m_fdWriteSockets );
		int nRes = select( int(m_sockMain)+1, NULL, &m_fdWriteSockets, NULL, &m_tvTimeOut );

		if( SOCKET_ERROR == nRes ) LOG_ERROR3_BREAK( szErrorFailedWithCodeFmt, "select", WSAGetLastError() );
		if( !nRes ) LOG_ERROR_BREAK( szSocketNotReady );
		if( !FD_ISSET(m_sockMain, &m_fdWriteSockets) ) LOG_ERROR_BREAK( "Timeout elapsed" );
	}
	while(false);

	return ( nRes > 0 );
}

bool CSocket::SendString( const char* pszData )
{
	return Send( pszData, (long)strlen(pszData) );
}

bool CSocket::SendString( const string& sValue )
{
	return Send( sValue.c_str(), (long)sValue.length() );
}

bool CSocket::RecvString( char* pszData, long nLen )
{
	bool bResult = Recv( pszData, nLen-1 );
	if( bResult ) pszData[m_nBytesRead] = 0;
	return bResult;
}

bool CSocket::RecvString( string& sValue )
{
	sValue.clear();
	char szReceiveBuffer[MAX_RETRIEVED_DATA_LEN];

	return RecvString( szReceiveBuffer, MAX_RETRIEVED_DATA_LEN - 1 ) 
		&& ( sValue.assign( szReceiveBuffer, m_nBytesRead ).length() != 0 );
}

ushort CSocket::RemotePort() const
{
	ushort nResult = 0;

	SOCKADDR saPeer;
	int nPeerLen = sizeof( saPeer );
	if( !getpeername(m_sockMain, &saPeer, &nPeerLen) ) {
		nResult = ntohs(((SOCKADDR_IN *)&saPeer)->sin_port);
	}

	return nResult;
}

char* CSocket::RemoteAddr() const
{
	char* pszResult = (char*)szEmptyAddress;

	SOCKADDR saPeer;
	int nPeerLen  = sizeof(saPeer);	
	if( !getpeername(m_sockMain, &saPeer, &nPeerLen) ) {
		pszResult = inet_ntoa(((SOCKADDR_IN *)&saPeer)->sin_addr);
	}

	return pszResult;
}

ushort CSocket::LocalPort() const
{
	ushort nResult = 0;

	SOCKADDR saSock;
	int nSockLen  = sizeof( saSock );	
	if( !getsockname(m_sockMain, &saSock, &nSockLen) ) {
		nResult = ntohs(((SOCKADDR_IN *)&saSock)->sin_port);
	}

	return nResult;
}

char* CSocket::LocalAddr() const
{
	char* pszResult = (char*)szEmptyAddress;

	SOCKADDR saSock;
	int nSockLen  = sizeof(saSock);	
	if( !getsockname(m_sockMain, &saSock, &nSockLen) ) {
		pszResult = inet_ntoa(((SOCKADDR_IN *)&saSock)->sin_addr);
	}

	return pszResult;
}

SOCKET& CSocket::GetSocket()
{
	return m_sockMain;
}

bool CSocket::Shutdown( int nHow /* = SD_BOTH */ )
{
	bool bResult = true;
	if( shutdown(m_sockMain, nHow) ) {
		LOG_ERROR3( szErrorFailedWithCodeFmt, "shutdown", WSAGetLastError() );
		bResult = false;
	}

	m_bConnected = false;
	return bResult;
}

void CSocket::SetRwTimeout( size_t nRwTimeout )
{
	m_nRwTimeOut = nRwTimeout;
}

size_t CSocket::GetRwTimeout()
{
	return m_nRwTimeOut;
}

size_t CSocket::GetWaitNextTimeout()
{
	return m_nWaitNextTimeOut;
}

void CSocket::SetWaitNextTimeout( size_t nLingerTimeout )
{
	m_nWaitNextTimeOut = nLingerTimeout;
}

void CSocket::SetLingerTimeout( size_t nLingerTimeout )
{
	m_nLingerTimeout = nLingerTimeout;
}

size_t CSocket::GetLingerTimeout()
{
	return m_nLingerTimeout;
}

bool& CSocket::GetUseHeaderLen()
{
	return m_bUseHeaderLength;
}

string CSocket::GetIpByName( const char* pszName )
{
	string sResult;
	hostent* pHost = gethostbyname(pszName);
	if( !WSAGetLastError() )  {
		sResult.assign( inet_ntoa(*(LPIN_ADDR)*pHost->h_addr_list) );
	}
	return sResult;
}

void CSocket::SetUseHeaderLen( bool bUse /* = false */ )
{
	m_bUseHeaderLength = bUse;
}

bool CSocket::IsConnected() const
{
	return m_bConnected;
}

/*static */
bool CSocket::CheckConnectionDropped()
{
	ulong ulLastError = WSAGetLastError();

	bool bResult = ulLastError == 10053 || ulLastError == 10054;
	return bResult;
}

size_t CSocket::GetBytesRead() const
{
	return m_nBytesRead;
}

size_t CSocket::GetBytesWritten() const
{
	return m_nBytesWritten;
}

//////////////////////////////////////////////
// CSocketHeadLenChanger implementation

CSocketHeadLenChanger::CSocketHeadLenChanger(CSocket *psSocket, bool bSetUseHeaderLen /*= false*/):
	m_psSocket(psSocket),
	m_bSetUseHeaderLen(bSetUseHeaderLen)
{
	ASSERTE(psSocket);
	psSocket->SetUseHeaderLen(m_bSetUseHeaderLen);
}

CSocketHeadLenChanger::~CSocketHeadLenChanger()
{
	m_psSocket->SetUseHeaderLen(!m_bSetUseHeaderLen);
}
