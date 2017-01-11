//#include "StdAfx.h"
#include "mysql_connection.h"

#include <common/asserts.h>
#include <threading/threading.h>


/////////////////////////////////////////////////////
// CMySQLRow implementation

const char* CMySQLRow::operator[]( size_t idx ) const
{
    static const char szEmptyValue[] = "";
    return ( m_msRow && m_nFields > idx ) ? m_msRow[ idx ] : szEmptyValue;
};


//--------------------------------------------------------------------------------------------------
// CMySQLResult implementation
CMySQLResult::CMySQLResult( ) : m_mrResult( MYSQL_REQYEST_FAILED ), m_prsRows( )
{
    m_prsRows = new ( std::nothrow ) CRowsStorage( );
    ASSERT( m_prsRows );
}

//--------------------------------------------------------------------------------------------------

CMySQLResult::CMySQLResult( CMySQLResult& mrObject ) : m_mrResult( ), m_prsRows( )
{
    swap( m_prsRows, mrObject.m_prsRows );
    swap( m_mrResult, mrObject.m_mrResult );
}

//--------------------------------------------------------------------------------------------------

CMySQLResult::CMySQLResult( CMySQLResult&& mrObject ) : m_mrResult( ), m_prsRows( )
{
    swap( m_prsRows, mrObject.m_prsRows );
    swap( m_mrResult, mrObject.m_mrResult );
}

//--------------------------------------------------------------------------------------------------

CMySQLResult& CMySQLResult::operator=( CMySQLResult& mrObject )
{
    swap( m_prsRows, mrObject.m_prsRows );
    swap( m_mrResult, mrObject.m_mrResult );

    return *this;
}

//--------------------------------------------------------------------------------------------------

CMySQLResult::~CMySQLResult( )
{
    if ( m_mrResult && MYSQL_REQYEST_FAILED != m_mrResult )
    {
        mysql_free_result( m_mrResult );
        m_mrResult = NULL;
    }

    FREE_POINTER( m_prsRows );
}

//--------------------------------------------------------------------------------------------------

size_t CMySQLResult::NumRows( ) const
{
    return ( m_mrResult ? (size_t)mysql_num_rows( m_mrResult ) : 0 );
}

//--------------------------------------------------------------------------------------------------

size_t CMySQLResult::NumFields( ) const
{
    return ( m_mrResult ? (size_t)mysql_num_fields( m_mrResult ) : 0 );
}

//--------------------------------------------------------------------------------------------------

size_t CMySQLResult::FetchAllRows( )
{
    if ( m_mrResult == nullptr )
    {
        return 0;
    }

    m_prsRows->clear( );

    MYSQL_ROW msRow = nullptr;
    while ( ( msRow = mysql_fetch_row( m_mrResult ) ) )
    {
        m_prsRows->push_back( msRow );
    }

    return m_prsRows->size( );
}

//--------------------------------------------------------------------------------------------------

char* CMySQLResult::GetCellValue( size_t nRow /*= 0*/, size_t nCol /*= 0*/ ) const
{
    return  nRow < m_prsRows->size( ) && nCol < NumFields( )
        ? ( *m_prsRows )[ nRow ][ nCol ] : nullptr;
}

//--------------------------------------------------------------------------------------------------

bool CMySQLResult::CheckValuesExist( size_t nFieldCount, size_t nRow /*= 0*/ ) const
{
    bool bResult = false;

    do
    {
        if ( nRow >= m_prsRows->size( ) ) break;

        size_t nFieldsCount = NumFields( );
        if ( nFieldCount > nFieldsCount ) break;

        bool bErrorOccurred = false;
        for ( size_t idx = 0; idx < nFieldCount; ++idx )
        {
            if ( NULL == ( *m_prsRows )[ nRow ][ idx ] )
            {
                bErrorOccurred = true;
                break;
            }
        }

        if ( bErrorOccurred ) break;
        bResult = true;
    }
    while ( false );

    return bResult;
}

//--------------------------------------------------------------------------------------------------
// CMySQLConnection implementation

CMySQLConnection::CMySQLConnection( )
    : m_msConnection( nullptr )
    , m_mrResult( nullptr )
    , m_bIsConnected( )
{
    m_msConnection = mysql_init( nullptr );
}

//--------------------------------------------------------------------------------------------------


CMySQLConnection::~CMySQLConnection( )
{
    Close( );
}

//--------------------------------------------------------------------------------------------------

bool CMySQLConnection::Connect( const char* szUser, const char* szPassword,
    const char* szDatabase, const char* szHost, uint16 nPort )
{
    if ( m_msConnection )
    {
        m_bIsConnected = nullptr != mysql_real_connect( m_msConnection, szHost, szUser, szPassword,
            szDatabase, nPort, nullptr, 0 );
    }

    return m_bIsConnected;
}

//--------------------------------------------------------------------------------------------------

bool CMySQLConnection::Query( const std::string& sQuery, CMySQLResult& msResult )
{
    do
    {
        msResult.m_mrResult = MYSQL_REQYEST_FAILED;
        if ( sQuery.empty( ) ) break;

        {
            static mutex_t mutex;
            mutex_locker_t lock( mutex );
            if ( 0 != mysql_query( m_msConnection, sQuery.c_str( ) ) ) break;
        }

        msResult.m_mrResult = mysql_store_result( m_msConnection );
        msResult.FetchAllRows( );
    }
    while ( false );

    return MYSQL_REQYEST_FAILED != msResult.m_mrResult;
}

//--------------------------------------------------------------------------------------------------

CMySQLResult CMySQLConnection::Query( const char* szQuery )
{
    CMySQLResult msResult;

    do
    {
        if ( !szQuery ) break;
        msResult.m_mrResult = MYSQL_REQYEST_FAILED;

        {
            static mutex_t mutex;
            mutex_locker_t lock( mutex );
            if ( 0 != mysql_query( m_msConnection, szQuery ) ) break;
        }

        msResult.m_mrResult = mysql_store_result( m_msConnection );
        msResult.FetchAllRows( );
    }
    while ( false );

    return msResult;
}

//--------------------------------------------------------------------------------------------------

CMySQLResult CMySQLConnection::Query( const std::string& sQuery )
{
    CMySQLResult msResult;
    Query( sQuery, msResult );
    return msResult;
}

//--------------------------------------------------------------------------------------------------

size_t CMySQLConnection::LastInsertId( )
{
    return (size_t)mysql_insert_id( m_msConnection );
}

//--------------------------------------------------------------------------------------------------

void CMySQLConnection::FreeResult( )
{
    if ( m_mrResult )
    {
        mysql_free_result( m_mrResult );
        m_mrResult = nullptr;
    }
}

//--------------------------------------------------------------------------------------------------

void CMySQLConnection::Close( )
{
    if ( m_msConnection )
    {
        mysql_close( m_msConnection );
        m_msConnection = nullptr;
    }
}

//--------------------------------------------------------------------------------------------------

bool CMySQLConnection::FetchRow( MYSQL_ROW& msRow )
{
    return m_mrResult && ( msRow = mysql_fetch_row( m_mrResult ) );
}

//--------------------------------------------------------------------------------------------------

bool CMySQLConnection::FetchRow( CMySQLRow& msrRow )
{
    bool bResult = FetchRow( msrRow.m_msRow );
    if ( bResult ) msrRow.m_nFields = NumFields( );
    return bResult;
}

//--------------------------------------------------------------------------------------------------

bool CMySQLConnection::GetResultValue( std::string& sValue, size_t nColumn /*= 0*/, size_t nRow /*= 0*/ )
{
    bool bResult = false;

    do
    {
        if ( nRow >= NumRows( ) ) break;
        if ( nColumn >= NumFields( ) ) break;

        bool error = false;
        MYSQL_ROW msRow = nullptr;
        for ( size_t idx = 0; idx <= nRow; ++idx )
        {
            if ( !FetchRow( msRow ) || !msRow )
            {
                error = true;
                break;
            }
        }

        if ( error ) break;

        sValue = msRow[ nColumn ];
        bResult = true;
    }
    while ( false );

    return bResult;
}

//--------------------------------------------------------------------------------------------------

size_t CMySQLConnection::AffectedRows( )
{
    return (size_t)mysql_affected_rows( m_msConnection );
}

//--------------------------------------------------------------------------------------------------

size_t CMySQLConnection::NumRows( )
{
    if ( !m_mrResult )
    {
        return 0;
    }

    return (size_t)mysql_num_rows( m_mrResult );
}

//--------------------------------------------------------------------------------------------------

size_t CMySQLConnection::NumFields( )
{
    if ( !m_mrResult )
    {
        return 0;
    }

    return (size_t)mysql_num_fields( m_mrResult );
}

//--------------------------------------------------------------------------------------------------

bool CMySQLConnection::SelectDB( const char* szDBName )
{
    return mysql_select_db( m_msConnection, szDBName ) == 0;
}

//--------------------------------------------------------------------------------------------------

