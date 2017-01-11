#ifndef __H_HTTPREQUEST__
#define __H_HTTPREQUEST__

#include "types.h"
#include "socket.h"

#define DEFAULT_HTTP_PORT 80
#define PAGE_SIZE_LIMIT 1024 * 1024 // 1 Mbyte
#define RECV_BUFFER_LENGTH 0xFFFF

#define HTTP_INVALID_RANGE 416
#define HTTP_STATUS_OK	200

extern const char szEndOfHeaders[];
extern const char szRowSeparator[];

extern const char szHeaderContentLength[];
extern const char szHeaderRange[];
extern const char szHeaderReferer[];
	extern const char szReferer[];
extern const char szHeaderTransferEncoding[];
	extern const char szTransferEncodingChunked[];

extern const char szHttpStatus[];
extern const char szLoadFrom[];

extern const char szErrorWithFile[];
extern const char szErrorOpenFile[];
extern const char szErrorHttp[];
extern const char szErrorInvalidHeaders[];
extern const char szErrorInvalidURL[];
extern const char szErrorConnection[];

class CHttpRequest
{
public:
	CHttpRequest();
	~CHttpRequest();

	bool Connect( const char* szServer, ushort nPort = DEFAULT_HTTP_PORT );
	bool Close();

	bool SendGetRequest( const char* szUrl );
	bool SendPostRequest( const char* szUrl, const char* szData, size_t nDataSize );
	bool SendPostRequest( const char* szUrl, const char* szData );

	bool RecvResponsePart( string& sResponse );
	size_t RecvResponsePart( char* szBuffer, size_t nBufferSize, bool bFirstPart = false );

	bool ExecutePostRequest( const char* szURL, const char* szData, string& sResponse, const char* szHeadersString = NULL );
	bool ExecuteGetRequest( const char* szURL, string& sResponse, const char* szHeadersString = NULL );

	bool DownloadFile( const char* szURL, const char* szFileName, size_t nStartFrom = 0 );
	bool LoadPage( const char* szUrl, string& sContent, size_t nPageSizeLimit = PAGE_SIZE_LIMIT );

	void AddHeader( const char* szName, const char* szValue );
	bool AddHeader( const char* szHeaderString );
	void AddHeaders( const char* szHeaderStrings );

	void AddCookie( const char* szCookie );

	void RemoveHeader( const char* szName );

	const char* GetHeadersString();
	bool CanRead( size_t nTimeout );

private:
	CSocket	m_sock;
	string m_sServer;
	map_ss	m_mssHeader;
};

#endif // __H_HTTPREQUEST__
