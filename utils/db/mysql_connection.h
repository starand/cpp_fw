#pragma once

#include <my_global.h>
#include <mysql.h>

#ifdef min
#undef min
#endif

#ifdef max
#undef max
#endif

#include <string>
#include <vector>


#define MYSQL_DEFAULT_PORT		3306
#define MYSQL_REQYEST_FAILED	((MYSQL_RES*)-1)


/////////////////////////////////////////////////////
// CMySQLRow declaration

class CMySQLRow
{
    friend class CMySQLConnection;
    friend class CMySQLResult;

public:
    CMySQLRow( ) : m_msRow( ), m_nFields( )
    {
    }

    bool empty( ) const
    {
        return ( NULL == m_msRow );
    }
    size_t size( ) const
    {
        return m_nFields;
    }

    const char* operator[]( size_t idx ) const;

private:
    MYSQL_ROW	m_msRow;
    size_t		m_nFields;
};


/////////////////////////////////////////////////////
// CMySQLResult declaration

typedef std::vector< MYSQL_ROW > CRowsStorage;

class CMySQLResult
{
    friend class CMySQLConnection;
public:
    CMySQLResult( );
    CMySQLResult( CMySQLResult& mrObject );
    CMySQLResult( CMySQLResult&& mrObject );
    ~CMySQLResult( );

    CMySQLResult& operator=( CMySQLResult& mrObject );

    size_t NumRows( ) const;
    size_t NumFields( ) const;

    size_t GetRowCount( ) const
    {
        return m_prsRows ? m_prsRows->size( ) : 0;
    }
    char* GetCellValue( size_t nRow = 0, size_t nCol = 0 ) const;
    bool CheckValuesExist( size_t nFieldCount, size_t nRow = 0 ) const;

    bool QueryFailed( ) const
    {
        return ( MYSQL_REQYEST_FAILED == m_mrResult );
    }

protected:
    size_t FetchAllRows( );

private:
    MYSQL_RES*		m_mrResult;
    CRowsStorage*	m_prsRows;
};


/////////////////////////////////////////////////////
// CMySQLConnection declaration

class CMySQLConnection
{
public:
    CMySQLConnection( );
    virtual ~CMySQLConnection( );

    bool IsConnected( ) const
    {
        return m_bIsConnected;
    }

    bool Connect( const char* szUser, const char* szPassword, const char* szDatabase,
        const char* szHost = "localhost", uint16 nPort = 3306 );
    void Close( );

    bool SelectDB( const char* szDBName );

    bool Query( const std::string& sQuery, CMySQLResult& msResult );
    CMySQLResult Query( const char* szQuery );
    CMySQLResult Query( const std::string& sQuery );
    size_t LastInsertId( );
    void FreeResult( );

    size_t AffectedRows( );
    size_t NumRows( );
    size_t NumFields( );

    bool FetchRow( MYSQL_ROW& msRow );
    bool FetchRow( CMySQLRow& msrRow );
    bool GetResultValue( std::string& sValue, size_t nColumn = 0, size_t nRow = 0 );

    MYSQL* GetHandle( ) const
    {
        return m_msConnection;
    }

protected:
    MYSQL*		m_msConnection;
    MYSQL_RES*	m_mrResult;
    std::string		m_sLastQuery;
    bool		m_bIsConnected;
};


