#ifndef __H_NOTIFY__
#define __H_NOTIFY__

#define DEFAULT_NOTIFIER_PORT 8000
#define DEFAUL_NOTIFIER_SERVER "127.0.0.1"

#define NOTIFY_NONE		0
#define NOTIFY_INFO		1
#define NOTIFY_WARNING	2
#define NOTIFY_ERROR	3
#define NOTIFY_USER		4

static const char szSeparator[] = "<@>";

class CNotify
{
public:
	CNotify( const string& sServerAddress = DEFAUL_NOTIFIER_SERVER, ushort nPort = DEFAULT_NOTIFIER_PORT );
	~CNotify();

	bool ShowMessage( const string& sMsg, const string& sTitle, uint nType = NOTIFY_INFO );
	bool ShowPackedMessage( const string& sMessageData );

	static bool UnpackMessage( const string& sFullMsg, string& sMsg, string& sTitle, uint& nType );

protected:
	string& PackMessage( const string& sMsg, const string& sTitle, uint nType );

private:
	ushort	m_nPort;
	string	m_sServerAddress;

};

#endif // __H_NOTIFY__
