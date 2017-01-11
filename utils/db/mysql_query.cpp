//#include <StdAfx.h>
#include "mysql_query.h"
#include "mysql_connection.h"

#include <common/asserts.h>
#include <common/macroes.h>
#include <common/logger.h>


#define CHECK_CONNECTION() \
    if( !m_pDBConn || !m_pDBConn->IsConnected() ) LOG_ERROR_BREAK( szNotConnected );

//////////////////////////////////////////////////
// mysql_query_t implementation

mysql_query_t::mysql_query_t()
    : m_pDBConn()
{
}

mysql_query_t::~mysql_query_t()
{
    FREE_POINTER( m_pDBConn );
}

bool mysql_query_t::init( const std::string& sUserName, const std::string& sPassword, const std::string& sDBName, const std::string& sHost /*= "localhost"*/, ushort nPort /*= MYSQL_DEFAULT_PORT*/ )
{
    START_FUNCTION_BOOL();

    static const char szUnableToConnectTODB[] = "Unable to connect to %s@%s:%u";

    if( m_pDBConn )
    {
        LOG_ERROR( "Mysql connection alread initialized" );
        break;
    }

    m_pDBConn = new (nothrow) CMySQLConnection();
    ASSERT( m_pDBConn != nullptr );

    if( !m_pDBConn->Connect(sUserName.c_str(), sPassword.c_str(), sDBName.c_str(), sHost.c_str(), nPort) )
    {
        LOG_ERROR( szUnableToConnectTODB, sUserName.c_str(), sHost.c_str(), nPort );
        break;
    }

    END_FUNCTION_BOOL();
}

bool mysql_query_t::execute( const std::string& sQuery, CMySQLResult& msResult ) const
{
    bool bSucceded = m_pDBConn->Query( sQuery, msResult );
    if( !bSucceded )
    {
        LOG_ERROR( "Unable to execute query: %s", sQuery.c_str() );
    }
    return bSucceded;
}

bool mysql_query_t::execute( const std::string& sQuery ) const
{
    return !m_pDBConn->Query(sQuery).QueryFailed();
}

