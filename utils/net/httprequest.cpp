#include "StdAfx.h"
#include "httprequest.h"
#include "strutils.h"
#include "fileutils.h"
#include "macroes.h"
#include "logerror.h"

using namespace StrUtils;

static char szRecvBuffer[ RECV_BUFFER_LENGTH ] = { 0 };

const char szHeaderSeparator[] = ": ";
const char szRowSeparator[] = "\r\n";
const char szEndOfHeaders[] = "\r\n\r\n";
const char szEmptyString[] = "";

const char szGetHeader[] = "GET %s HTTP/1.1%s";
const char szGetHeaderFull[] = "GET %s HTTP/1.1%s%s%s";
const char szPostHeader[] = "POST %s HTTP/1.1%s";
const char szPostHeaderFull[] = "POST %s HTTP/1.1%s%s%s%s";
const char szReferer[] = "http://%s/";
const char szLoadFrom[] = "bytes=%lld-";

const char szHttpStatus[] = "HTTP/1.";
const char szHeaderHost[] = "Host";
const char szHeaderContentLength[] = "Content-Length";
const char szHeaderCookie[] = "Cookie";
const char szHeaderRange[] = "Range";
const char szHeaderReferer[] = "Referer";
const char szHeaderTransferEncoding[] = "Transfer-Encoding";
	const char szTransferEncodingChunked[] = "chunked";

const char szHeaderContentType[] = "Content-Type";
	const char szContentTypeFormData[] = "application/x-www-form-urlencoded";

const char szErrorEmptyInputData[] = "Empty input data";
const char szErrorNotConnected[] = "Not connected";
const char szErrorWithFile[] = "File error";
const char szErrorOpenFile[] = "Can not open file : %s";
const char szErrorHttp[]  = "HTTP error : %s";
const char szErrorInvalidHeaders[] = "Incorrect HTTP headers";
const char szErrorInvalidURL[] = "Incorrect URL : %s";
const char szErrorConnection[] = "Can not connect to %s:%u";
const char szErrorFileNotFound[] = "File %s not found";

//////////////////////////////////////////////////
// CHttpRequest implementation

CHttpRequest::CHttpRequest()
{
	// m_sock.SetWaitNextTimeout( );
}

CHttpRequest::~CHttpRequest()
{
	Close();
}

bool CHttpRequest::Connect( const char* szServer, ushort nPort )
{
	m_sServer = szServer;
	AddHeader( szHeaderHost, szServer );
	return m_sock.Connect( szServer, nPort );
}

bool CHttpRequest::Close()
{
	return m_sock.Close();
}

bool CHttpRequest::SendGetRequest( const char* szUrl )
{
	bool bResult = false;
	do
	{
		if( !szUrl || *szUrl == 0 )	LOG_ERROR_BREAK( szErrorEmptyInputData );
		if( !m_sock.IsConnected() ) LOG_ERROR_BREAK( szErrorNotConnected );

		string sGetData = FormatString( szGetHeaderFull, szUrl, szRowSeparator, GetHeadersString(), szRowSeparator );
		if( !m_sock.Send(sGetData.c_str(), sGetData.length()) ) break;

		bResult = true;
	}
	while( false );
	return bResult;
}

bool CHttpRequest::SendPostRequest( const char* szUrl, const char* szData, size_t nDataSize )
{
	bool bResult = false;
	do
	{

		bResult = true;
	}
	while( false );
	return bResult;
}

bool CHttpRequest::SendPostRequest( const char* szUrl, const char* szData )
{
	bool bResult = false;
	do
	{
		if( !szUrl || *szUrl == 0 )	LOG_ERROR_BREAK( szErrorEmptyInputData );
		if( !m_sock.IsConnected() ) LOG_ERROR_BREAK( szErrorNotConnected );
		if( !szData ) szData = szEmptyString;

		AddHeader( szHeaderContentType, szContentTypeFormData );
		AddHeader( szHeaderContentLength, IntToStr(strlen(szData)) );
		string sPostData = FormatString( szPostHeaderFull, szUrl, szRowSeparator, GetHeadersString(), szRowSeparator, szData );
		if( !m_sock.Send(sPostData.c_str(), sPostData.length()) ) break;

		bResult = true;
	}
	while( false );
	return bResult;
}

bool CHttpRequest::RecvResponsePart( string& sResponse )
{
	size_t nBytesReceived = 0;
	if( m_sock.Recv(szRecvBuffer, RECV_BUFFER_LENGTH - 1) )
	{
		nBytesReceived = m_sock.GetBytesRead();
		sResponse.assign( szRecvBuffer, nBytesReceived );
	}
	return ( nBytesReceived > 0 );
}

bool CHttpRequest::CanRead( size_t nTimeout )
{
	return m_sock.CanRead( nTimeout );
}

size_t CHttpRequest::RecvResponsePart( char* szBuffer, size_t nBufferSize, bool bFirstPart /*= false*/ )
{
	if( !m_sock.Recv(szBuffer, nBufferSize - 1, bFirstPart) ) return 0;
	return m_sock.GetBytesRead();
}

bool CHttpRequest::ExecutePostRequest( const char* szURL, const char* szData, string& sResponse, const char* szHeadersString )
{
	bool bResult = false;
	do
	{
		if( !szURL || !szURL[0] ) LOG_ERROR_BREAK( szErrorEmptyInputData );
		if( !m_sock.IsConnected() ) LOG_ERROR_BREAK( szErrorNotConnected );
		if( !szData ) szData = szEmptyString;

		if( szHeadersString ) AddHeaders( szHeadersString );

		AddHeader( szHeaderContentLength, IntToStr(strlen(szData)) );
		string sPostData = FormatString( szPostHeaderFull, szURL, szRowSeparator, GetHeadersString(), szRowSeparator, szData );

		if( !m_sock.Send(sPostData.c_str(), sPostData.length()) ) break;
		if( !m_sock.Recv(szRecvBuffer, RECV_BUFFER_LENGTH - 1) ) break;

		sResponse.assign( szRecvBuffer );
		bResult = true;
	}
	while( false );
	return bResult;
}

bool CHttpRequest::ExecuteGetRequest( const char* szURL, string& sResponse, const char* szHeadersString )
{
	bool bResult = false;
	do
	{
		if( !szURL || !szURL[0] ) LOG_ERROR_BREAK( szErrorEmptyInputData );
		if( !m_sock.IsConnected() ) LOG_ERROR_BREAK( szErrorNotConnected );

		if( szHeadersString ) AddHeaders( szHeadersString );

		string sGetData = FormatString( szGetHeaderFull, szURL, szRowSeparator, GetHeadersString(), szRowSeparator );

		if( !m_sock.Send(sGetData.c_str(), sGetData.length()) ) break;
		if( !m_sock.Recv(szRecvBuffer, RECV_BUFFER_LENGTH - 1) ) break;

		sResponse.assign( szRecvBuffer );
		bResult = true;
	}
	while( false );
	return bResult;
}

bool CHttpRequest::DownloadFile( const char* szURL, const char* szFileName, size_t nStartFrom /*= 0*/ )
{
	bool bResult = false;
	
	do
	{
		if( !szURL || !szFileName || *szURL == 0 || *szFileName == 0 ) LOG_ERROR_BREAK( szErrorEmptyInputData );
		if( !m_sock.IsConnected() ) LOG_ERROR_BREAK( szErrorNotConnected );

		if( nStartFrom )
		{
			AddHeader( szHeaderRange, FormatString("bytes=%u-", nStartFrom) );
			AddHeader( szHeaderReferer, FormatString("http://%s/", m_sServer.c_str()) );
		}
		
		string sGetData = FormatString( szGetHeaderFull, szURL, szRowSeparator, GetHeadersString(), szRowSeparator );
		if( !m_sock.Send(sGetData.c_str(), sGetData.length()) ) LOG_ERROR_BREAK("Send error!");

		FILE* fp = NULL;
		if( fopen_s(&fp, szFileName, "wb") && !fp ) LOG_ERROR_BREAK( szErrorWithFile );

		bool bError = false;
		size_t nContentLength = 0;
		bool bHtppHeaders = true;

		do
		{
			bError = true;
			char* szWriteData = szRecvBuffer;
			size_t nBytesRead = 0;
			if( !m_sock.Recv(szRecvBuffer, RECV_BUFFER_LENGTH - 1) || 
				!( nBytesRead = m_sock.GetBytesRead()) )  LOG_ERROR_BREAK( "Recv error" );

			if( bHtppHeaders )
			{
				bHtppHeaders = false;
				char* szPos  = strstr( szRecvBuffer, szEndOfHeaders );
				if( !szPos )
				{
					LOG_ERROR_BREAK( szErrorInvalidHeaders );
				}
				*szPos = 0;

				char* szContentLength = strstr( szRecvBuffer, szHeaderContentLength );
				if( !szContentLength ) LOG_ERROR2_BREAK( szErrorFileNotFound, szURL );

				szContentLength += strlen( szHeaderContentLength );
				sscanf_s( szContentLength, "%u", &nContentLength );
				
				szPos += strlen( szEndOfHeaders );
				size_t nHeadersSize = szPos - szRecvBuffer;
				if( nHeadersSize > nBytesRead )
				{
					LOG_ERROR_BREAK( szErrorInvalidHeaders );
				}

				szWriteData += nHeadersSize;
				nBytesRead -= nHeadersSize;
			}

			if( nBytesRead != fwrite(szWriteData, 1, nBytesRead, fp) ) LOG_ERROR_BREAK( szErrorWithFile );
			
			nContentLength -= nBytesRead;
			bError = false;
		}
		while( nContentLength );

		if( fp ) fclose(fp);
		if( !bError ) bResult = true;
	}
	while( false );

	return bResult;
}

bool CHttpRequest::LoadPage( const char* szUrl, string& sContent, size_t nPageSizeLimit )
{
	bool bResult = false;

	do
	{
		sContent.clear();
		if( !szUrl || *szUrl == 0 ) LOG_ERROR_BREAK( szErrorEmptyInputData );
		if( !m_sock.IsConnected() ) LOG_ERROR_BREAK( szErrorNotConnected );

		string sGetData = FormatString( szGetHeaderFull, szUrl, szRowSeparator, GetHeadersString(), szRowSeparator );
		if( !m_sock.Send(sGetData.c_str(), sGetData.length()) ) break;

		bool bError = false, bHtppHeaders = true, bSkipRest = false;
		while( m_sock.CanRead(500) )
		{
			bError = true;

			char* szWriteData = szRecvBuffer;
			size_t nBytesRead = 0;
			if( !m_sock.Recv(szRecvBuffer, RECV_BUFFER_LENGTH - 1) || 
				!( nBytesRead = m_sock.GetBytesRead()) ) break;
			szWriteData[ nBytesRead ] = 0;

			if( bHtppHeaders )
			{
				bHtppHeaders = false;
				char* szPos  = strstr( szRecvBuffer, szEndOfHeaders );
				if( !szPos ) 
				{
					LOG_ERROR_BREAK( szErrorInvalidHeaders );
				}
				szPos += strlen( szEndOfHeaders );

				size_t nHeadersSize = szPos - szRecvBuffer;
				if( nHeadersSize > nBytesRead )
				{
					LOG_ERROR_BREAK( szErrorInvalidHeaders );
				}

				szWriteData += nHeadersSize;
				nBytesRead -= nHeadersSize;
			}

			bError = false;

			if( !bSkipRest ) sContent.append( szWriteData );
			if( sContent.length() > nPageSizeLimit ) bSkipRest = true;
		}

		if( bError ) break;
		bResult = true;
	}
	while( false );
	
	return bResult;
}

void CHttpRequest::AddHeader( const char* szName, const char* szValue )
{
	m_mssHeader[ szName ] = szValue;
}

bool CHttpRequest::AddHeader( const char* szHeaderString )
{
	bool bResult = false;
	do
	{
		string sName, sValue;
		if( !ParseNameValuePair(szHeaderString, sName, sValue, ":") ) break;
		Trim( sValue );
		if( sName.empty() || sValue.empty() ) break;

		m_mssHeader[ sName ] = sValue;
		bResult = true;
	}
	while( false );
	return bResult;
}

void CHttpRequest::AddHeaders( const char* szHeaderStrings )
{
	if( szHeaderStrings && *szHeaderStrings )
	{
		map_ss mssHeaders;
		split_map( szHeaderStrings, mssHeaders, "\n", ": " );

		for( map_ss::iterator mapIt = mssHeaders.begin(); mapIt != mssHeaders.end(); ++mapIt )
		{
			string sName = mapIt->first, sVal = mapIt->second;
			Trim( sName ); Trim( sVal );
			if( !sName.empty() && !sVal.empty() ) m_mssHeader[ sName ] = sVal;
		}
	}
}

const char* CHttpRequest::GetHeadersString()
{
	static string sResult;

	sResult.clear();
	size_t nHeadersCount = m_mssHeader.size();
	for( map_ss::iterator mapIt = m_mssHeader.begin(); mapIt != m_mssHeader.end(); ++mapIt )
	{
		sResult += mapIt->first + szHeaderSeparator + mapIt->second + szRowSeparator;
	}

	return sResult.c_str();
}

void CHttpRequest::RemoveHeader( const char* szName )
{
	map_ss::iterator mapIt = m_mssHeader.find( szName );
	if( mapIt != m_mssHeader.end() ) m_mssHeader.erase( mapIt );
}

void CHttpRequest::AddCookie( const char* szCookie )
{
	if( szCookie && *szCookie ) m_mssHeader[ szHeaderCookie ] = szCookie;
}


