#include "StdAfx.h"
#include "httputils.h"
#include "httprequest.h"
#include "fileutils.h"
#include "strutils.h"
#include "logerror.h"
#include "algorithm.h"

const char chDblQuote = '\"';
const char chSlash = '/';
const char szDblSlash[] = "//";
const char szHttpPrefix[] = "http://";
const size_t nHttpPrefixLen = strlen( szHttpPrefix );

namespace HttpUtils {

bool DownloadFile( const char* szUrl, const char* szResultFile, bool bResume, ProgressCallBack pcbFunction )
{
	int64 nFileSize = 0, nBytesDownloaded = 0;

	do
	{
		ushort nPort;
		string sHost, sPage;
		if( !StrUtils::ParseUrl(szUrl, sHost, nPort, sPage) ) LOG_ERROR2_BREAK( szErrorInvalidURL, szUrl);

		CHttpRequest httpRequest;
		if( !httpRequest.Connect(sHost.c_str(), nPort) ) LOG_ERROR3_BREAK( szErrorConnection, sHost.c_str(), nPort );

		uint64 nStartFrom = 0;
		if( bResume )
		{
			FileUtils::GetFileSize( szResultFile, nStartFrom, "rb" );

			httpRequest.AddHeader( szHeaderRange, StrUtils::FormatString(szLoadFrom, nStartFrom) );
			httpRequest.AddHeader( szHeaderReferer, StrUtils::FormatString(szReferer, sHost.c_str() ) );
		}

		if( !httpRequest.SendGetRequest(sPage.c_str()) ) break;

		FILE* fp = NULL;
		if( fopen_s(&fp, szResultFile, ( bResume ? "ab" : "wb")) ) LOG_ERROR2_BREAK( szErrorOpenFile, szResultFile );

		bool bHeaderPacket = true;
		size_t nBytesReceived = 0;
		static char szBuffer[ RECV_BUFFER_LENGTH ] = { 0 };
		int nPercents = -1;

		do
		{
			const char* szData = szBuffer;
			nBytesReceived = httpRequest.RecvResponsePart( szBuffer, RECV_BUFFER_LENGTH-1 );
			if( !nBytesReceived ) LOG_ERROR_BREAK( "\rDownloading error " );

			if( bHeaderPacket )
			{
				bHeaderPacket = false;
				// if file is completely downloaded - skip downloading
				if( !(bResume && HTTP_INVALID_RANGE == GetHttpStatus(szBuffer)) ) 
				{
					char* szContentLength = (char*)strstr( szBuffer, szHeaderContentLength );
					if( !szContentLength ) LOG_ERROR_BREAK( szErrorInvalidHeaders );
					szContentLength += strlen( szHeaderContentLength ) + strlen( szRowSeparator );
					sscanf_s( szContentLength, "%llu", &nFileSize );
				}

				nFileSize += nStartFrom;
				nBytesDownloaded += nStartFrom;

				szData = strstr( szBuffer, szEndOfHeaders );
				if( !szData ) LOG_ERROR_BREAK( szErrorInvalidHeaders );
				szData += strlen( szEndOfHeaders );
				nBytesReceived -= szData - szBuffer;
			}

			if( nBytesReceived != fwrite(szData, 1, nBytesReceived, fp) ) LOG_ERROR_BREAK( szErrorWithFile );

			nBytesDownloaded += nBytesReceived;
			if( pcbFunction && nPercents != (nBytesDownloaded * 100 / nFileSize) )
			{
				(*pcbFunction)( nFileSize, nBytesDownloaded, PathUtils::GetShortFileName(szResultFile) );
				nPercents = (int)(nBytesDownloaded * 100 / nFileSize);
			}
		}
		while( nBytesDownloaded < nFileSize );
		httpRequest.Close();

		if( fp ) fclose( fp );
	}
	while( false );

	return ( nBytesDownloaded == nFileSize );
}

bool DownloadPage( const char* szUrl, string& sContent, size_t nPageSizeLimit )
{
	bool bResult = false;

	static char szBuffer[ RECV_BUFFER_LENGTH ] = { 0 };

	do
	{
		ushort nPort;
		string sHost, sPage;
		if( !StrUtils::ParseUrl(szUrl, sHost, nPort, sPage) ) LOG_ERROR2_BREAK( szErrorInvalidURL, szUrl);

		CHttpRequest httpRequest;
		if( !httpRequest.Connect(sHost.c_str(), nPort) ) LOG_ERROR3_BREAK( szErrorConnection, sHost.c_str(), nPort );
		if( !httpRequest.SendGetRequest(sPage.c_str()) ) break;

		bool bChunked = false;
		bool bHeaderPacket = true;
		size_t nTotalBytesRead = 0;
		size_t nContentLength = string::npos;

		do
		{
			char* szData = szBuffer;
			size_t nBytesRead = httpRequest.RecvResponsePart( szBuffer, RECV_BUFFER_LENGTH, bHeaderPacket );
			if( !nBytesRead ) break;
			
			if( bHeaderPacket ) // first part
			{
				bHeaderPacket = false;
				if( HTTP_STATUS_OK != GetHttpStatus(szBuffer) )
					LOG_ERROR2_BREAK( szErrorHttp, "Status not found" );

				map_ss mssHeaders;
				if( !GetHeadersAsMap(szBuffer, mssHeaders) )
					LOG_ERROR_BREAK( szErrorInvalidHeaders );

				if( mssHeaders[ szHeaderTransferEncoding ] == szTransferEncodingChunked ) bChunked = true;
				
				// get Content-Length header
				const char* szContentLength = mssHeaders[ szHeaderContentLength ].c_str();
				if( szContentLength && strlen(szContentLength) ) {
					nContentLength = atol( szContentLength );
				}

				szData = strstr( szBuffer, szEndOfHeaders );
				if( !szData ) LOG_ERROR_BREAK( szErrorInvalidHeaders );
				szData += strlen( szEndOfHeaders );
				nBytesRead -= szData - szBuffer;
			}

			nTotalBytesRead += nBytesRead;
			sContent.append( szData, nBytesRead );

			if( string::npos != nContentLength && nTotalBytesRead == nContentLength ) break;

			const char* szEnd = sContent.c_str() + sContent.length() - 7;
			if( memcmp(szEnd, "\r\n0\r\n\r\n", 7) == 0 ) break;
		}
		while( true );

		if( bChunked && !DeleteChunckSizes(sContent) ) LOG_ERROR2_BREAK( szErrorHttp, " Invalid chunks");

		if( sContent.empty() && nContentLength != 0 ) LOG_ERROR_BREAK( "Empty response" );
		bResult = true;
	}
	while( false );

	return bResult;
}

bool DownloadPage(const string& sUrl, string& sContent, size_t nPageSizeLimit /*= PAGE_SIZE_LIMIT*/ )
{
	ASSERTE(!sUrl.empty());
	return DownloadPage( sUrl.c_str(), sContent, nPageSizeLimit );
}

bool DeleteChunckSizes( string& sContent )
{
	bool bResult = false;

	size_t nPos = 0, nEndRowPos = 0, nChunckSize = 0;
	static int nRowSepLen = strlen( szRowSeparator );

	while( nPos < sContent.length() )
	{
		nEndRowPos = sContent.find( szRowSeparator, nPos );
		if( string::npos == nEndRowPos ) break;
		sscanf_s( sContent.c_str() + nPos, "%x", &nChunckSize );

		if( !nChunckSize )
		{
			sContent.erase( nPos );
			//cout << sContent.c_str() + nPos;
			bResult = true;
			break;
		}

		size_t nLen = nEndRowPos + nRowSepLen - nPos;
		sContent.erase( nPos, nLen );
		nPos += nChunckSize;
		if( nPos >= sContent.length() ) break;
		sContent.erase( nPos, nRowSepLen );
	}

	return bResult;
}

bool SeparateHeadersAndData( string& sData, string& sHeaders )
{
	sHeaders.clear();
	
	size_t nHeadersEndPos = sData.find( szEndOfHeaders );
	bool bResult = ( string::npos != nHeadersEndPos );
	
	if( bResult )
	{
		sHeaders = sData.substr( 0, nHeadersEndPos );
		sData.erase( 0, nHeadersEndPos + strlen(szEndOfHeaders) );
	}

	return bResult;
}

size_t GetHeadersAsMap( const char* szData, map_ss& mssResult )
{
	mssResult.clear();
	if( const char* szHeadersEnd = strstr(szData, szEndOfHeaders) )
	{
		string sHeaders( szData, szHeadersEnd );
		StrUtils::split_map( sHeaders.c_str(), mssResult, "\n", ":", true );
	}

	return mssResult.size();
}

size_t GetHttpStatus( const char* szData )
{
	size_t nResult = 0;
	do
	{
		size_t nStatusLen = strlen( szHttpStatus ) + 2;
		char* szPos = (char*)strstr( szData, szHttpStatus );
		if( !szPos || strlen(szPos) <= nStatusLen ) break;
		szPos += nStatusLen;
		sscanf_s( szPos, "%i", &nResult );
	}
	while( false );
	return nResult;
}

size_t ParseLinks( const string& sPageContent, string_v& vsLinks )
{
	static const int nHrefLen = 4;
	static const char szHrefAttr[] = "href";
	static const char szCorrectLinkStartChars[] = "hH./";

	size_t nPos = 0, nEndPos = 0;
	while( string::npos != (nPos = sPageContent.find(szHrefAttr, nPos)) )
	{
		nPos += nHrefLen;
		if( sPageContent[nPos] != '=' && sPageContent[++nPos] != '=' ) continue;
		// start and end quotes
		if( string::npos == (nPos = sPageContent.find(chDblQuote, nPos)) ) break;
		if( string::npos == (nEndPos = sPageContent.find(chDblQuote, ++nPos)) ) break;
		
		string sLink = sPageContent.substr(nPos, nEndPos-nPos);
		if( strchr(szCorrectLinkStartChars, sLink[0]) ) vsLinks.push_back( sLink );

		nPos = ++nEndPos;
	}
	return vsLinks.size();
}

string CutServerFromLink( const string& sLink )
{
	string sResult;
	static const char szServerEndChars[] = "/#?\\";
	do
	{
		size_t nPos = sLink.find( szDblSlash ); 
		if( string::npos == nPos ) break;

		nPos += strlen( szDblSlash );
		nPos = sLink.find_first_of( szServerEndChars, nPos );
		if( string::npos == nPos ) sResult = sLink + chSlash;
		else sResult = sLink.substr( 0, nPos ) + chSlash;
	}
	while( false );
	return sResult;
}

string& HttpEncode( string& sUrlString )
{
	static const char szInvalidURLChars[] = " ;=$,<>^\'\\[]{}|\"";
	static const char szReplaceCharCode[] = "\x20\x3b\x3D\x26\x2C\x3C\x3E\x5E\x60\x5c\x5B\x5D\x7B\x7D\x7C\x22";
	static const size_t nInvalidCharsCount = ARRAY_SIZE( szInvalidURLChars );

	static const sizeint siBufferSize = 8;
	static char szBuffer[siBufferSize] = { 0 };

	size_t nPos = 0, nUrlLength = sUrlString.length();
	for( size_t idx = 0; idx < nUrlLength; ++idx )
	{
		if( string::npos != (nPos = Algorithm::GetValuePos(szInvalidURLChars, nInvalidCharsCount, sUrlString[idx])) )
		{
			sUrlString.erase( idx, 1 );
			sprintf_s( szBuffer, siBufferSize, "%%%2.2X", szReplaceCharCode[nPos] );
			sUrlString.insert( idx, szBuffer );
			idx += 2;
		}
	}

	return sUrlString;
}

};
