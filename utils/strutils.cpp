#include "StdAfx.h"

#include "strutils.h"
#include "macroes.h"
#include "asserts.h"

#include <stdlib.h>
#include <stdarg.h>
#include <cstring>

#ifdef LINUX
#	define stricmp strcasecmp
#else
#	pragma warning(disable : 4996)
#	define snprintf sprintf_s
#endif


#define DEFAULT_HTTP_PORT	80
#define DELIMS " \t\r\n\x9"

#define LEFT_ANGLE		'<'
#define RIGHT_ANGLE		'>'

#ifndef DEBUG_MAX_BUFFER_SIZE
#	define DEBUG_MAX_BUFFER_SIZE 10240 // 10 Kbytes
#endif

static const char g_szHexDigits[] = "0123456789ABCDEFabcdef";
static const sizeint g_siHeDigitsCount = strlen(g_szHexDigits);

static const char szCommandSeparators[] = " \t";

namespace StrUtils {

char* FormatString(const char* szFormat, ...)
{
	char* pszResult = NULL;

	static char szOutputString[ DEBUG_MAX_BUFFER_SIZE ] = {};

	va_list vaArgs;

	try
	{
		va_start(vaArgs, szFormat);
		vsnprintf(szOutputString, DEBUG_MAX_BUFFER_SIZE, szFormat, vaArgs);
		va_end(vaArgs);
		pszResult = szOutputString;
	}
	catch(...)
	{
		pszResult = (char*)"";
	}

	return pszResult;
}

// format string
bool FormatString(string& sResult, const char* szFormat, ...)
{
	bool bResult = true;

	try
	{
		char szOutputString[ DEBUG_MAX_BUFFER_SIZE ] = {};

		va_list vaArgs;
		va_start( vaArgs, szFormat );
		vsnprintf( szOutputString, DEBUG_MAX_BUFFER_SIZE, szFormat, vaArgs );
		va_end( vaArgs );
		sResult.assign( szOutputString );
	}
	catch(...)
	{
		bResult = false;
	}
	return bResult;
}

// format string
bool FormatString(string& sResult, const char* szFormat, va_list vaArgs)
{
	bool bResult = true;

	try
	{
		char szOutputString[ DEBUG_MAX_BUFFER_SIZE ] = {};

		vsnprintf(szOutputString, DEBUG_MAX_BUFFER_SIZE, szFormat, vaArgs);
		sResult.assign(szOutputString);
	}
	catch(...)
	{
		bResult = false;
	}
	return bResult;
}

bool IsNumber(const string& sValue)
{
	bool bRes = true;
	for(size_t i = 0; i < sValue.length(); i++)
	{
		if(!isdigit(sValue[i]))
		{
			bRes = false;
			break;
		}
	}
	return bRes;
}

bool IsIpAddr(const string& sIpAddr, IpData* pvDigits /* = NULL */)
{
	bool bRes =  false;
	do
	{
		bool bBreak = false;
		string sNextPart = sIpAddr + '.';
		for(int i = 0; i < 4; i++)
		{
			size_t nPos = sNextPart.find(".");
			if(nPos == string::npos)
			{
				bBreak = true;
				break;
			}

			string sDigit = sNextPart.substr(0, nPos);
			//cout  << sDigit << endl;
			if(sDigit.empty() || !IsNumber(sDigit))
			{
				bBreak = true;
				break;
			}
			sNextPart.erase(0, nPos+1);
			if(pvDigits)
			{
				pvDigits->push_back(atoi(sDigit.c_str()));
			}
		}
		if(!sNextPart.empty() || bBreak) break;
		bRes = true;
	}
	while(false);
	return bRes;
}

const char* MacAddr( const uchar* pMac )
{
	const size_t nBufLen = 18;
	static char szBuffer[ nBufLen ] = { 0 };

	snprintf(szBuffer, nBufLen, "%2.2X:%2.2X:%2.2X:%2.2X:%2.2X:%2.2X",
		pMac[0]&0xFF, pMac[1]&0xFF, pMac[2]&0xFF, pMac[3]&0xFF, pMac[4]&0xFF, pMac[5]&0xFF);

	return szBuffer;
}

const char* IpAddr( const uchar* pszIp )
{
	const size_t nBufLen = 16;
	static char szBuffer[nBufLen] = {0};

	snprintf(szBuffer, nBufLen, "%u.%u.%u.%u",
		pszIp[0], pszIp[1], pszIp[2], pszIp[3]);

	return szBuffer;
}

string& Trim(string& sText)
{
	size_t nStartPos = 0;
	// find leading spaces
	nStartPos = sText.find_first_not_of( DELIMS );
	if(string::npos == nStartPos) sText.clear();
	else
	{
		// trim spaces
		size_t nEndPos = sText.find_last_not_of( DELIMS );
		if( nEndPos != string::npos ) nEndPos = nEndPos - nStartPos + 1;
		sText = sText.substr( nStartPos, nEndPos );
	}

	return sText;
}

// deleting comment from end of string
template<class T> string& DeleteBefore(string& sProperty, const T& delim)
{
	size_t nPos = sProperty.find_first_of(delim);
	if(nPos != string::npos)
	{
		sProperty.erase(nPos);
	}
	return sProperty;
}

size_t Explode(string sInput, string sDelims, string_v& vsResult)
{
	size_t nPos, nCount = 0;
	string sItem;
	vsResult.clear();
	while((nPos = sInput.find_first_of(sDelims)) != string::npos)
	{
		sItem = sInput.substr(0, nPos);
		sItem = Trim(sItem);
		if(!sItem.empty())
		{
			vsResult.push_back(sItem);
			nCount++;
		}
		sInput.erase(0, nPos+1);
	}
	sItem = Trim(sInput);
	if(!sItem.empty())
	{
		vsResult.push_back(sItem);
		nCount++;
	}
	return nCount;
}

string Implode(const string_v& vsInput, string sDelim)
{
	string sResult;
	size_t nCount = vsInput.size();
	if(nCount)
	{
		string_v::const_iterator svIt = vsInput.begin();
		while(svIt != vsInput.end())
		{
			sResult += *svIt;
			if(nCount > 1)
			{
				sResult += sDelim;
			}
			nCount--;
			++svIt;
		}
	}
	return sResult;
}

const char* IntToStr( int num )
{
	static const int nBufSize = 15;
	static char szBuffer[nBufSize] = { 0 };
	snprintf(szBuffer, nBufSize, "%i", num);
	return szBuffer;
}

string IntToStr(ushort usValue)
{
	const int siBufferSize = 10;
	char szBuffer[siBufferSize] = { 0 };

	snprintf(szBuffer, siBufferSize, "%d", usValue);
	return szBuffer;
}

string IntToStr(uint64 uiValue)
{
	const int siBufferSize = 20;
	char szBuffer[siBufferSize] = { 0 };
	snprintf(szBuffer, siBufferSize, "%lld", uiValue);

	return szBuffer;
}

#if !__x86_64__
const char* IntToStr(size_t num)
{
	static const int nBufSize = 15;
	static char szBuffer[nBufSize] = { 0 };
    sprintf(szBuffer, "%u", num);

	return szBuffer;
}
#endif

string IntToHex(int nValue)
{
	const sizeint siBufferSize = 10;
	char szBuffer[10] = { siBufferSize };

	snprintf(szBuffer, siBufferSize, "%2.2x", nValue);
	return szBuffer;
}


string UShortToStr( ushort nValue )
{
	static const size_t nBufLen = 32;
	static string sResult( nBufLen, '0' );
	snprintf((char*)sResult.c_str(), nBufLen, "%u", nValue);
	return sResult;
}

// convert string to int
int StrToInt(const string& sValue)
{
	return atoi(sValue.c_str());
}

int StrToInt(char* pszValue)
{
	return atoi(pszValue);
}

ushort StrToUShort( const string& sValue )
{
	return (ushort)atoi( sValue.c_str() );
}

uint64 StrToUInt64(const char *szValue)
{
	ASSERTE(szValue);

	uint64 uiResult = 0;
	sscanf(szValue, "%lld", &uiResult);

	return uiResult;
}


bool IsQuote( char chSym )
{
	return ( chSym == '\"' || chSym == '\'' );
}

string& DelQuotes( string& sValue, bool bTrim /*= true*/ )
{
	if( bTrim ) Trim( sValue );

	size_t nLen = sValue.length();
	if( nLen )
	{
		if( nLen > 1 && IsQuote(sValue[nLen - 1]) ) sValue.erase( nLen - 1 );
		if( IsQuote(sValue[0]) ) sValue.erase( 0, 1 );
	}

	return sValue;
}

string& DelBothQuotes( string& sValue, bool bTrim /*= true*/ )
{
	if( bTrim ) Trim( sValue );

	size_t nLen = sValue.length();
	if( nLen >= 2 && IsQuote(sValue[0]) && sValue[nLen-1])
	{
		sValue.erase( nLen-1 );
		sValue.erase( 0, 1 );
	}

	return sValue;
}

string& DelAngleQuotes(string& sVarValue, bool bTrim /*= true*/)
{
	string sValue = sVarValue;
	if (bTrim)
	{
		Trim(sValue);
	}

	if (sValue[0] == LEFT_ANGLE)
	{
		sValue.erase(0, 1);
	}

	sizeint siLastPos = sValue.length() - 1;
	if (sValue[siLastPos] == RIGHT_ANGLE)
	{
		sValue.erase(siLastPos, 1);
	}

	sVarValue = sValue;
	return sVarValue;
}

char* StringToLower( char* szString )
{
	size_t nLength = szString ? strlen( szString ) : 0;
	for( size_t idx = 0; idx < nLength; ++idx )
	{
		szString[ idx ] = (char)tolower( szString[idx] );
	}
	return szString;
}

char* StringToUpper( char* szString )
{
	size_t nLength = szString ? strlen( szString ) : 0;
	for( size_t idx = 0; idx < nLength; ++idx )
	{
		szString[ idx ] = (char)toupper( szString[idx] );
	}
	return szString;
}

int CompareStrings(const char* pszFirst, const char* pszSecond)
{
	while( *pszFirst && *pszSecond && *pszFirst == *pszSecond )
	{
		++pszFirst;
		++pszSecond;
	}

	if( *pszFirst == *pszSecond ) return 0;
	else if( *pszFirst > *pszSecond) return 1;
	else return -1;

}

int CompareStrings(const char* pszFirst, const char* pszSecond, bool bIgnoreCase)
{
	if( bIgnoreCase )
	{
		while( *pszFirst && *pszSecond && tolower(*pszFirst) == tolower(*pszSecond) )
		{
			++pszFirst;
			++pszSecond;
		}

		char cFirst = tolower(*pszFirst), cSecond = tolower(*pszSecond);
		if( cFirst == cSecond ) return 0;
		else if( cFirst > cSecond) return 1;
		else return -1;
	}
	else
		return CompareStrings(pszFirst, pszSecond);
}

// change register to upper
string& ToUpper(string& sValue)
{
	size_t nValueLen = sValue.length();
	for(size_t i = 0; i < nValueLen; ++i)
	{
		sValue[i] = toupper(sValue[i]);
	}
	return sValue;
}

// change register to lower
string& ToLower(string& sValue)
{
	size_t nValueLen = sValue.length();
	for(size_t i = 0; i < nValueLen; ++i)
	{
		sValue[i] = tolower(sValue[i]);
	}
	return sValue;
}

int CompareStrings(const string& sFirst, const string& sSecond, bool bIgnoreCase /* = false */)
{
	return CompareStrings(sFirst.c_str(), sSecond.c_str(), bIgnoreCase);
}


// parse HTTP url address
bool ParseUrl(string sUrl, string& sHost, ushort& nPort, string& sPage)
{
	bool bResult = false;

	do
	{
		if( sUrl.empty() ) break;

		size_t nPos = sUrl.find("://");
		if( nPos != string::npos )
		{
			sUrl.erase(0, nPos + 3);
		}

		nPos = sUrl.find('/');
		if( nPos == string::npos ) sUrl.append("/");

		nPos = sUrl.find('/');
		sHost = sPage = sUrl;
		sHost.erase(nPos);
		sPage.erase(0, nPos);

		nPos = sHost.find(':');
		if( nPos != string::npos )
		{
			string sPort = sHost.substr(nPos + 1);
			int nTempPort = atoi(sPort.c_str());
			if(nTempPort) nPort = nTempPort;
			sHost.erase(nPos);
		}
		else
		{
			nPort = DEFAULT_HTTP_PORT;
		}

		bResult = true;
	}
	while (false);

	return bResult;
}

string AddSlashes( string sInput, const char szReplaceChars[] /* = "\"\'\\"*/ )
{
	size_t nPos = 0;
	while( string::npos != (nPos = sInput.find_first_of(szReplaceChars, nPos)) )
	{
		sInput.insert( nPos, "\\" );
		nPos += 2;
	}

	return sInput;
}

void RemoveBrackets( string& sValue, const char cBeginBracket /*= '('*/,
	const char cEndBracket /*= ')'*/ )
{
	if( sValue[0] == cBeginBracket )
	{
		sValue.erase( 0, 1 );
	}

	size_t nLastPos = sValue.length() - 1;
	if( sValue[nLastPos] == cEndBracket )
	{
		sValue.erase( nLastPos, 1 );
	}
}

bool ParseNameValuePair( const char* szInput, string& sTag, string& sValue,
	const char* szDelims /*= "="*/ )
{
	bool bResult = false;

	do
	{
		sTag.clear();
		sValue.clear();

		char* pDelimPos = NULL;
		if( !szInput || !strlen(szInput) ||
			!(pDelimPos = strpbrk((char*)szInput, szDelims)) ) break;

		sTag.assign( szInput, pDelimPos - szInput );
		sValue.assign( ++pDelimPos );

		Trim( sTag );
		Trim( sValue );

		bResult = true;
	}
	while(false);

	return bResult;
}

// find first of delim chars
size_t split( const char* szInput, vector<string>& vsResult, const char* szDelims /*= " \t"*/ )
{
	size_t nDelimLen = strlen( szDelims );
	char* szCurPos = const_cast<char*>( szInput );
	vsResult.clear();

	while( szCurPos )
	{
		char* szEndPos = strpbrk( szCurPos, szDelims );
		if( !szEndPos || szEndPos <= szCurPos ) break;

		string sValue( szCurPos, szEndPos);
		Trim( sValue );

		if( !sValue.empty() )
		{
			vsResult.push_back( sValue );
		}

		szCurPos = szEndPos + nDelimLen;
	}

	if( szCurPos && strlen(szCurPos) )
	{
		string sValue( szCurPos );
		Trim( sValue );
		if( !sValue.empty() )
		{
			vsResult.push_back( sValue );
		}
	}

	return vsResult.size();
}

// find delimiter - whole string
size_t split( string sInput, string_v& vsResult, const string& sDelim )
{
	size_t nPos = 0;

	const size_t nDelimLen = sDelim.length();
	ASSERT(nDelimLen > 0);

	sInput += sDelim;
	while( string::npos != (nPos = sInput.find(sDelim)) )
	{
		string sItem = sInput.substr( 0, nPos );
		if( !sItem.empty() ) vsResult.push_back( sItem );

		sInput.erase( 0, nPos + nDelimLen );
	}

	return vsResult.size();
}

size_t split_map( const char* szInput, map<string, string>& mssResult,
	const char* szLineDelims /*= "\n"*/, const char* szPairDelims /*= "="*/, bool bTrim/* = false*/ )
{
	mssResult.clear();

	string_v vsLines;
	size_t nLinesCount = split( szInput, vsLines, szLineDelims );
	if( nLinesCount )
	{
		for( size_t idx = 0; idx < nLinesCount; ++idx )
		{
			string sName, sValue;
			if( ParseNameValuePair(vsLines[idx].c_str(), sName, sValue, szPairDelims)
				&& !sName.empty() && !sValue.empty() )
			{
				if( bTrim ) { Trim( sName ); Trim( sValue ); }
				mssResult[ sName ] = sValue;
			}
		}
	}
	return mssResult.size();
}

void map_implode( const map_ss& mssMap, string& sResult, char cLineDelim /*= '\n'*/, char cPairDelim /*= '='*/ )
{
	sResult.clear();

	for( map_ss::const_iterator itMap = mssMap.begin(); itMap != mssMap.end(); ++itMap )
	{
		sResult += itMap->first + cPairDelim + itMap->second + cLineDelim;
	}
}

char* find_first_not_of( const char* szInput, const char* szDelims /*= " \t"*/ )
{
	char* szResult = NULL;
	if( szInput )
	{
		char* szPos = const_cast<char*>( szInput );
		while( *szPos && strchr(szDelims, *szPos) ) szPos++;
		if( *szPos ) szResult = szPos;
	}
	return szResult;
}

size_t CharVectorToStringVector( const char* pszInput[], string_v& vsOutput )
{
	vsOutput.clear();
	for( size_t idx = 0; pszInput[idx]; ++idx )
	{
		vsOutput.push_back( pszInput[idx] );
	}
	return vsOutput.size();
}

string& Replace( string& sInput, const char* szFrom, const char* szTo )
{
    size_t nPos = 0;
	while( string::npos != (nPos = sInput.find(szFrom, nPos)) )
	{
		sInput.replace( nPos, strlen(szFrom), szTo );
	}
	return sInput.assign( sInput.c_str() );
}

string& Replace( string& sValue, char cFrom, char cTo )
{
	for_each( sValue, nLen, i ) if( sValue[i] == cFrom ) sValue[i] = cTo;
	return sValue;
}

char* Replace( char* szValue, char cFrom, char cTo )
{
	if( !szValue || *szValue == 0 ) return NULL;
	char* szPos = szValue;
	while( *szPos )
	{
		if( *szPos == cFrom ) *szPos = cTo;
		++szPos;
	}
	return szValue;
}

size_t GetSubstringEntries( const string& sInput, const char* szSubString, uint_v& viResult )
{
	do
	{
		viResult.clear();
		if( !szSubString || !szSubString[0] ) break;

		size_t nPos = 0, nSubLen = strlen( szSubString );
		while( string::npos != (nPos = sInput.find(szSubString, nPos)) )
		{
			viResult.push_back( nPos );
			nPos += nSubLen;
		}
	}
	while ( false );
	return viResult.size();
}

string& ClearSpaces( string& sString, const char* szSpaces )
{
	size_t nPos = 0;
	while( nPos < sString.length() )
	{
		if( strchr(szSpaces, sString[nPos]) ) sString.erase( nPos, 1 );
		else ++nPos;
	}
	return sString;
}

const char* DeleteSpaces( string& sValue, const char* szSpaces /*= "\r\n\t"*/ )
{
	size_t nLength = sValue.length(), nPos = 0;
	while( nPos < nLength )
	{
		if( !strchr(szSpaces, sValue[nPos]) )
		{
			++nPos;
			continue;
		}

		sValue.erase( nPos, 1 );
		--nLength;
	}

	return sValue.c_str();
}

size_t ParseCommandLine( const char* szCommandLine, string_v& vsParams )
{
	do
	{
		bool bInQuotes = false;
		if( !szCommandLine ) break;

		size_t nPrevPos = 0;
		size_t nLen = strlen( szCommandLine );
		for( size_t idx = 0; idx < nLen; ++idx )
		{
			if( szCommandLine[idx] == '\"' ) bInQuotes = !bInQuotes;
			if( !bInQuotes && (szCommandLine[idx] == ' ' || szCommandLine[idx] == '\t') )
			{
				vsParams.push_back( string(szCommandLine+nPrevPos, szCommandLine+idx) );
				nPrevPos = idx + 1;
			}
		}

		vsParams.push_back( szCommandLine + nPrevPos );
	}
	while( false );

	return vsParams.size();
}

string ParseCommandParams( const string& sCommandLine, string& sParams )
{
	string sResult;
	do
	{
		size_t nStartPos = sCommandLine.find_first_not_of( szCommandSeparators );
		if( string::npos == nStartPos ) break;

		size_t nEndPos = (sCommandLine[nStartPos] == '\"')
			? sCommandLine.find('\"' ) : sCommandLine.find_first_of( szCommandSeparators );
		if( string::npos == nEndPos )
		{
			sResult = sCommandLine.substr( nStartPos );
			sParams.clear();
		}
		else
		{
			sResult = sCommandLine.substr( nStartPos, nEndPos-nStartPos );
			sParams = sCommandLine.substr( nEndPos + 1 );
			Trim( sParams );
		}
	}
	while( false );

	return sResult;
}

string LongToBitString( uint nValue )
{
	static const size_t nBitCount = sizeof( uint ) * BITSINBYTE;
	static const char szOne[] = "1";
	static const char szZero[] = "0";

	string sResult;
	for( size_t idx = 0; idx < nBitCount; ++idx )
	{
		sResult.insert( 0, (nValue & 1) ? szOne : szZero );
		nValue >>= 1;
	}

	return sResult;
}

int CompareString(const char *szFirstValue, const char *szSecondValue, bool bCaseSensitive/* = false*/)
{
	return bCaseSensitive
		? strcmp(szFirstValue, szSecondValue)
		: stricmp(szFirstValue, szSecondValue);
}

bool CompareStr(const char *szFirstValue, const char *szSecondValue, bool bCaseSensitive)
{
	return CompareString(szFirstValue, szSecondValue, bCaseSensitive) == 0;
}

#ifdef _ALGORITHM_
bool IsHexDigit(char chChar)
{
	bool bResult = std::binary_search(g_szHexDigits, g_szHexDigits + g_siHeDigitsCount, chChar);
	return bResult;
}
#endif

} // namespace StrUtils

