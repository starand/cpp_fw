#ifndef __XSOCKET_H_INCLUDED
#define __XSOCKET_H_INCLUDED

#include "types.h"
#ifdef WINDOWS
#	include <winsock2.h>
#else
#	include <netinet/in.h>
//#	define SD_BOTH SHUT_RDWR
#	define SD_BOTH 2
typedef int SOCKET;

void ignore_sigpipe( );
#endif


#define MAX_CLIENT_QUEUE_SIZE	5



enum ESOCKETERRROR
{
    SE__MIN,
    SE_NO_ERROR = SE__MIN,

    SE_UNRESOLVED_NAME,
    SE_INVALID_IP,

    SE_CREATE_SOCKET_ERROR,
    SE_CONNECTION_ERROR,
    SE_BIND_ERROR,
    SE_LISTEN_ERROR,

    SE__MAX,
};


const char *get_error_description( ESOCKETERRROR error );


class socket_t
{
public:
    socket_t( );
    ~socket_t( );

public:
    static int get_last_error( );
    static string get_ip_by_name( const string& sHostName );
    static bool ip_valid( const string& ip );

public:
    bool connect( const string& sHostName, ushort usPort );

    bool listen( ushort usPort, long lAddress = INADDR_ANY, int nMaxClient = MAX_CLIENT_QUEUE_SIZE );
    bool listen( ushort usPort, const char* szAddress, int nMaxClient /*= MAXCONN*/ );
    socket_t *accept( ) const;

    bool shutdown( int nMode = SD_BOTH );
    bool close( );

    long send( void *data, int nLength );
    long recv( void *buffer, int nLength );

public:
    ushort get_local_port( ) const;
    ushort get_remote_port( ) const;

    const char *get_local_address( ) const;
    const char *get_remote_address( ) const;

    uint get_raddr( ) const;
    uint get_laddr( ) const;

    string get_local_endpoint( ) const;
    string get_remote_endpoint( ) const;

    bool is_connected( ) const;
    ESOCKETERRROR get_error( ) const;

    SOCKET get( ) const
    {
        return m_socket;
    }

private:
    struct sockaddr_in	m_saLocalAddr;
    struct sockaddr_in	m_saRemoteAddr;
    SOCKET				m_socket;
    bool				m_bConnected;

    ESOCKETERRROR		m_error;
};

#endif // __XSOCKET_H_INCLUDED
