#ifndef __H_HTTPUTILS__
#define __H_HTTPUTILS__

#include "types.h"
#include "httprequest.h"

namespace HttpUtils
{
	typedef void (*ProgressCallBack)( int64, int64, const char* );

	bool DownloadFile( const char* szUrl, const char* szResultFile, 
		bool bResume = false, ProgressCallBack pcbFunction = NULL );

	bool DownloadPage( const char* szUrl, string& sContent, size_t nPageSizeLimit = PAGE_SIZE_LIMIT );
	bool DownloadPage(const string& sUrl, string& sContent, size_t nPageSizeLimit = PAGE_SIZE_LIMIT );
	bool DeleteChunckSizes( string& sContent );

	bool SeparateHeadersAndData( string& sData, string& sHeaders );
	size_t GetHeadersAsMap( const char* szData, map_ss& mssResult );
	size_t GetHttpStatus( const char* szData );

	size_t ParseLinks( const string& sPageContent, string_v& vsLinks );
	string CutServerFromLink( const string& sLink );

	string& HttpEncode( string& sUrlString );
};

#endif // __H_HTTPUTILS__
