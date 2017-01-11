#ifndef __SOCKETS__
#define __SOCKETS__

#include "types.h"
#include <winsock2.h>


#define MAXCONN			5

#define DEFAULT_RW_TIMEOUT			100		// ms
#define DEFAULT_WAITNEXT_TIMEOUT	100		// ms

#define DEFAULT_LINGER_TIMEOUT		5		// sec

#define DEFAULT_CHECK_ATTEMPTS		3

#define USE_HEADER_LEN	true

class CSocket
{
public:

#ifdef SEND_LEN_PREIFX
	CSocket( bool bUseLength = true );
#else
	CSocket( bool bUseLength = false );
#endif

	~CSocket();

	bool Connect( const string& sHostName, short nPort );
	bool Close();

	bool Listen( ushort nPort, long lAddr = INADDR_ANY, int nMaxClient = MAXCONN );
	bool Listen( ushort nPort, const char* szAddress, int nMaxClient = MAXCONN );
	CSocket* Accept();
	
	bool Send( const char* pszData, long nLen );
	bool Recv( char* pszData, long nLen, bool bEmptyError = true );
	bool CanRead( long nMillisec = 0 );
	bool CanWrite( long nMillisec = 0 );

	bool SendString( const char* pszData );
	bool SendString( const string& sValue );
	bool RecvString( char* pszData, long nLen );
	bool RecvString( string& sValue );

	bool Shutdown( int nHow = SD_BOTH );

	char* LocalAddr() const;
	char* RemoteAddr() const;
	ushort LocalPort() const;
	ushort RemotePort() const;

	size_t GetRwTimeout();
	void SetRwTimeout( size_t nRwTimeout );
	size_t GetWaitNextTimeout();
	void SetWaitNextTimeout( size_t nLingerTimeout );
	size_t GetLingerTimeout();
	void SetLingerTimeout( size_t nLingerTimeout );

	bool& GetUseHeaderLen();
	void SetUseHeaderLen( bool bUse = false );

	static string GetIpByName( const char* pszName );

	bool IsConnected() const;
	static bool CheckConnectionDropped();

	size_t GetBytesRead() const;
	size_t GetBytesWritten() const;

protected:
	bool InitSockets();
	SOCKET& GetSocket();

	bool _recv( char* pszData, long nLen );

private:
	static WSADATA		m_wsaData;
	static bool			m_bInitialized;

	SOCKET				m_sockMain;
	sockaddr_in			m_saLocalAddr;
	SOCKET				m_sockClient;
	sockaddr_in			m_saRemoteAddr;
	int					m_RSALen;
	
	fd_set				m_fdReadSockets;
	fd_set				m_fdWriteSockets;
	timeval				m_tvTimeOut;

	size_t				m_nRwTimeOut;
	size_t				m_nWaitNextTimeOut;
	size_t				m_nLingerTimeout;

	size_t				m_nBytesWritten;
	size_t				m_nBytesRead;
	size_t				m_nCheckAttempts;

	bool				m_bUseHeaderLength;
	bool				m_bClientMode;
	bool				m_bConnected;
};


class CSocketHeadLenChanger
{
public:
	CSocketHeadLenChanger(CSocket *psSocket, bool bSetUseHeaderLen = false);
	~CSocketHeadLenChanger();

private:
	CSocket		*m_psSocket;
	bool		m_bSetUseHeaderLen;

};

typedef CSocket socket_t;

#endif // __SOCKETS__
