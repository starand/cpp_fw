#include "StdAfx.h"
#include "xsocket.h"

#include "platform.h"
#include "asserts.h"


//#define SOCKET_DEBUG
#ifdef SOCKET_DEBUG
#	include <iomanip>
#endif


static const char g_szEmptyAddress[] = "";


#ifdef WINDOWS
#	include <Ws2tcpip.h>
#	pragma warning(disable : 4996)
#	pragma comment(lib, "ws2_32.lib")
typedef int socklen_t;

bool g_bWsaInitialized;

void init_socket( )
{
    if ( !g_bWsaInitialized )
    {
        WSADATA wdData;

        int nResult = WSAStartup( MAKEWORD( 2, 2 ), &wdData );
        ASSERTE( nResult == 0 );

        g_bWsaInitialized = true;
    }
}

#else // LINUX
//#	include <sys/types.h>
#	include <unistd.h>
#	include <sys/socket.h>
#	include <errno.h>
#	include <signal.h>
#	include <string.h>
#	include <netdb.h>
#	include <arpa/inet.h>
#	include <errno.h>

#	define SOCKET_ERROR (-1)

enum
{
    INVALID_SOCKET = -1
};

void init_socket( )
{
}

void ignore_sigpipe( )
{
    signal( SIGPIPE, SIG_IGN );
}

int closesocket( SOCKET socket )
{
    shutdown( socket, SHUT_RDWR );
    return ::close( socket );
}
#endif


const char *g_aszSocketErrorArray[ SE__MAX ] =
{
    "",	//SE_NO_ERROR,

    "Unable to resolve name",			//SE_UNRESOLVED_NAME,
    "Incorrect ip address",				//SE_INVALID_IP,

    "Unable to create socket",			//SE_CREATE_SOCKET_ERROR,
    "Unable to establish connection",	//SE_CONNECTION_ERROR,
    "Unable to bind port",				//SE_BIND_ERROR,
    "Unable to listen port",			//SE_LISTEN_ERROR,
};


const char *get_error_description( ESOCKETERRROR error )
{
    ASSERT( IN_RANGE( error, SE__MIN, SE__MAX ) );
    return g_aszSocketErrorArray[ error ];
}

socket_t::socket_t( )
    : m_socket( 0 )
    , m_bConnected( false )
    , m_error( SE_NO_ERROR )
{
    init_socket( );

    memset( &m_saLocalAddr, 0, sizeof( m_saLocalAddr ) );
    memset( &m_saRemoteAddr, 0, sizeof( m_saRemoteAddr ) );
}

socket_t::~socket_t( )
{
}


/*static */
int socket_t::get_last_error( )
{
#ifdef WINDOWS
    return WSAGetLastError( );
#else
    return errno;
#endif
}

/*static */
string socket_t::get_ip_by_name( const string& sHostName )
{
    ASSERTE( !sHostName.empty( ) );
    hostent* pHost = gethostbyname( sHostName.c_str( ) );

    if ( pHost )
    {
        return inet_ntoa( *(in_addr *)*pHost->h_addr_list );
    }
    else if ( ip_valid( sHostName ) )
    {
        return sHostName;
    }
    else
    {
        LOG_DEBUG( "Invalid host address : %s", sHostName.c_str( ) );
        return "";
    }
}

/*static */
bool socket_t::ip_valid( const string& ip )
{
    return inet_addr( ip.c_str( ) ) != INADDR_NONE;
}


bool socket_t::connect( const string& sHostName, ushort usPort )
{
    ASSERTE( !sHostName.empty( ) && usPort > 0 );
    bool bResult = false;

    do
    {
        string sAddress = get_ip_by_name( sHostName );
        if ( sAddress.empty( ) )
        {
            m_error = SE_UNRESOLVED_NAME;
            return false;
        }

        m_saLocalAddr.sin_family = AF_INET;
        m_saLocalAddr.sin_addr.s_addr = inet_addr( sAddress.c_str( ) );
        m_saLocalAddr.sin_port = htons( usPort );

        if ( INVALID_SOCKET == ( m_socket = socket( AF_INET, SOCK_STREAM, 0 ) ) )
        {
            m_error = SE_CREATE_SOCKET_ERROR;
            break;
        }

        if ( SOCKET_ERROR == ::connect( m_socket, (sockaddr *)&m_saLocalAddr, sizeof( m_saLocalAddr ) ) )
        {
            m_error = SE_CONNECTION_ERROR;
            break;
        }

        bResult = m_bConnected = true;
    }
    while ( false );

    return bResult;
}


bool socket_t::listen( ushort usPort, long lAddress /*= INADDR_ANY*/, int nMaxClient /*= MAX_CLIENT_QUEUE_SIZE*/ )
{
    ASSERTE( usPort > 0 );
    bool bResult = false;

    do
    {
        m_saLocalAddr.sin_addr.s_addr = htonl( lAddress );
        m_saLocalAddr.sin_port = htons( usPort );
        m_saLocalAddr.sin_family = AF_INET;

        if ( INVALID_SOCKET == ( m_socket = ::socket( AF_INET, SOCK_STREAM, 0 ) ) )
        {
            break;
        }

        int nOption = 1;
        if ( SOCKET_ERROR == ::setsockopt( m_socket, SOL_SOCKET, SO_REUSEADDR, (char *)&nOption, sizeof( nOption ) ) )
        {
            break;
        }

        if ( SOCKET_ERROR == ::bind( m_socket, ( struct sockaddr * )&m_saLocalAddr, sizeof( m_saLocalAddr ) ) )
        {
            break;
        }

        if ( SOCKET_ERROR == ::listen( m_socket, nMaxClient ) )
        {
            break;
        }

        bResult = true;
    }
    while ( false );

    return bResult;
}

bool socket_t::listen( ushort usPort, const char* szAddress, int nMaxClient /*= MAXCONN*/ )
{
    ASSERTE( szAddress && szAddress[ 0 ] );

    uint uAddress = ntohl( inet_addr( szAddress ) );
    return listen( usPort, uAddress, nMaxClient );
}

socket_t *socket_t::accept( ) const
{
    socket_t *psResultSocket = NULL;

    do
    {
        SOCKET clientSocket;
        socklen_t nAddressLen = sizeof( m_saRemoteAddr );
        if ( INVALID_SOCKET == ( clientSocket = ::accept( m_socket, ( struct sockaddr * )&m_saRemoteAddr, &nAddressLen ) ) )
        {
            break;
        }

        if ( !( psResultSocket = new (nothrow)socket_t( ) ) )
        {
            closesocket( clientSocket );
            break;
        }

        psResultSocket->m_socket = clientSocket;
        psResultSocket->m_saLocalAddr = m_saLocalAddr;
        psResultSocket->m_saRemoteAddr = m_saRemoteAddr;
        psResultSocket->m_bConnected = true;
    }
    while ( false );

    return psResultSocket;
}


bool socket_t::shutdown( int nMode /*= SD_BOTH*/ )
{
    m_bConnected = false;
    return ::shutdown( m_socket, nMode ) == 0;
}

bool socket_t::close( )
{
    int res = 0;

    if ( m_socket != INVALID_SOCKET )
    {
        res = ::closesocket( m_socket );
    }

    m_bConnected = false;
    return res == 0;
}



#ifdef SOCKET_DEBUG
#	include <iostream>
#	define DUMP_LEN 30

void dump_buffer( const char *buffer, int length )
{
    if ( length < 0 ) return;
    int len = length > DUMP_LEN ? DUMP_LEN : length;
    for ( int i = 0; i < len; ++i )
    {
        cout << ( hex ) << setw( 2 ) << setfill( '0' ) << (int)buffer[ i ] << ' ';
    }
    if ( length > len ) cout << "..";
    cout << endl;
}

#endif


long socket_t::send( void *data, int nLength )
{
    ASSERTE( data && nLength > 0 );

    long lBytesSend = 0; // assume the value 0 means error
    const char *data_ptr = (const char *)data;

    do
    {
        int nSendResult = ::send( m_socket, data_ptr, nLength, 0 );
#ifdef SOCKET_DEBUG
        cout << nSendResult << " bytes send -- ";
        dump_buffer( data_ptr, nSendResult );
#endif
        if ( SOCKET_ERROR == nSendResult )
        {
            int last_error = get_last_error( );
#ifdef WINDOWS
            if ( WSA_IO_PENDING != last_error )
#else // LINUX
            if ( EAGAIN != last_error )
#endif
            {
                //cout << "NOT WAS_IO_PENDING" << endl;
            }
#ifdef SOCKET_DEBUG
            else
            {
                cout << "WSA IOPENDING" << endl;
            }
#endif // SOCKET_DEBUG

            lBytesSend = 0;
            break;
        }

        lBytesSend += nSendResult;
        data_ptr += nSendResult;
        nLength -= nSendResult;
    }
    while ( nLength );

    return lBytesSend;
}

long socket_t::recv( void *buffer, int nLength )
{
    ASSERTE( buffer && nLength > 0 );

    long lBytesRecv = 0;
    char *buffer_ptr = (char *)buffer;

    while ( nLength )
    {
        int nRecvResult = ::recv( m_socket, buffer_ptr, nLength, 0 );
        if ( 0 == nRecvResult ) // connection was gracefully closed
        {
            return 0;
        }
        else if ( SOCKET_ERROR == nRecvResult )
        {
            return 0;
        }

        lBytesRecv += nRecvResult;
        buffer_ptr += nRecvResult;
        nLength -= nRecvResult;
    }
    while ( false );

    return lBytesRecv;
}


ushort socket_t::get_local_port( ) const
{
    sockaddr saPeer;
    socklen_t nPeerLen = sizeof( saPeer );
    return getsockname( m_socket, &saPeer, &nPeerLen ) ? 0 : ntohs( ( (sockaddr_in *)&saPeer )->sin_port );
}

ushort socket_t::get_remote_port( ) const
{
    ASSERTE( m_bConnected == true );
    sockaddr saPeer;
    socklen_t nPeerLen = sizeof( saPeer );
    return getpeername( m_socket, &saPeer, &nPeerLen ) ? 0 : ntohs( ( (sockaddr_in *)&saPeer )->sin_port );
}


const char *socket_t::get_local_address( ) const
{
    ASSERTE( m_socket );

    sockaddr saPeer;
    socklen_t nPeerLen = sizeof( saPeer );
    return getsockname( m_socket, &saPeer, &nPeerLen ) ? g_szEmptyAddress : inet_ntoa( ( (sockaddr_in *)&saPeer )->sin_addr );
}

const char *socket_t::get_remote_address( ) const
{
    ASSERTE( m_bConnected == true );

    sockaddr saPeer;
    socklen_t nPeerLen = sizeof( saPeer );
    return getpeername( m_socket, &saPeer, &nPeerLen ) ? g_szEmptyAddress : inet_ntoa( ( (sockaddr_in *)&saPeer )->sin_addr );
}


uint socket_t::get_raddr( ) const
{
    return (uint)m_saRemoteAddr.sin_addr.s_addr;
}

uint socket_t::get_laddr( ) const
{
    return (uint)m_saLocalAddr.sin_addr.s_addr;
}


string socket_t::get_local_endpoint( ) const
{
    char buffer[ 24 ];
    sprintf( buffer, "%s:%u", get_local_address( ), get_local_port( ) );
    return buffer;
}

string socket_t::get_remote_endpoint( ) const
{
    char buffer[ 24 ];
    sprintf( buffer, "%s:%u", get_remote_address( ), get_remote_port( ) );
    return buffer;
}


bool socket_t::is_connected( ) const
{
    return m_bConnected;
}

ESOCKETERRROR socket_t::get_error( ) const
{
    return m_error;
}
