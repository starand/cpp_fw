#ifndef __H_MYSQL_WRAPPER__
#define __H_MYSQL_WRAPPER__

#include <common/types.h>
#include <string>


#ifndef MYSQL_DEFAULT_PORT
#	define MYSQL_DEFAULT_PORT 3306
#endif


class CMySQLResult;
class CMySQLConnection;

class mysql_query_t
{
public:
	mysql_query_t();
	virtual ~mysql_query_t();

	virtual bool init( const std::string& user_name, const std::string& password,
                           const std::string& db_name, const std::string& host = "localhost",
                           ushort nPort = MYSQL_DEFAULT_PORT );
	bool execute( const std::string& query, CMySQLResult& msResult ) const;
	bool execute( const std::string& query ) const;

private:
	CMySQLConnection* m_pDBConn;
};

#endif // __H_MYSQL_WRAPPER__
