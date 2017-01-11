#ifndef _STR_UTILS_
#define _STR_UTILS_

#include <string>
#include <vector>
#include <map>
#include <sstream>
#include "types.h"

#ifdef WIN32
#	define stricmp _stricmp
#endif

namespace StrUtils
{
	typedef vector<int> IpData;

	char* FormatString(const char* szFormat, ...);
	bool FormatString(string& sResult, const char* szFormat, ...);
	bool FormatString(string& sResult, const char* szFormat, va_list vaArgs);

	bool IsNumber(const string& sValue);
	bool IsIpAddr(const string& sIpAddr, IpData* pvDigits = NULL);
	const char* MacAddr(const uchar* pszMac);
	const char* IpAddr(const uchar* pszIp);

	string& Trim(string& sText);
	string& Replace( string& sInput, const char* szFrom, const char* szTo );
	string& Replace( string& sValue, char cFrom, char cTo );
	char* Replace( char* szValue, char cFrom, char cTo );
	const char* DeleteSpaces( string& sValue, const char* szSpaces = "\r\n\t" );

	bool IsQuote( char chSym );
	string& DelQuotes( string& sValue, bool bTrim = true );
	string& DelBothQuotes( string& sValue, bool bTrim = true );
	string& DelAngleQuotes(string& sVarValue, bool bTrim = true);

	template<class T> string& DeleteBefore(string& sProperty, const T& delim);
	size_t Explode(string sInput, string sDelims, string_v& vsResult);
	string Implode(const string_v& vsInput, string sDelim);

	const char* IntToStr( int num );

#if !__x86_64__
	const char* IntToStr( size_t num );
#endif

	string IntToStr(ushort usValue);
	string IntToStr(uint64 uiValue);

	string IntToHex(int nValue);

	string UShortToStr( ushort nValue );

	int StrToInt(const string& sValue);
	int StrToInt(char* pszValue);

	ushort StrToUShort( const string& sValue );
	uint64 StrToUInt64(const char *szValue);
	inline uint64 StrToUInt64(const string& sValue) { return StrToUInt64(sValue.c_str()); }

	char* StringToLower( char* szString );
	char* StringToUpper( char* szString );
	int CompareStrings(const char* pszFirst, const char* pszSecond, bool bIgnoreCase);
	int CompareStrings(const char* pszFirst, const char* pszSecond, bool bIgnoreCase);

	string& ToUpper(string& sValue);
	string& ToLower(string& sValue);
	int CompareStrings(const string& sFirst, const string& sSecond, bool bIgnoreCase = false);

	bool ParseUrl(string sUrl, string& sHost, ushort& nPort, string& sPage);

	string AddSlashes( string sInput, const char szReplaceChars[] = "\"\'\\" );
	void RemoveBrackets( string& sValue, const char cBeginBracket = '(', 
		const char cEndBracket = ')' );

	bool ParseNameValuePair( const char* szInput, string& sTag, string& sValue,
		const char* szDelims = "=" );

	size_t split( const char* szInput, vector<string>& vsResult, const char* szDelims = " \t" );
	size_t split( string sInput, string_v& vsResult, const string& sDelim );
	size_t split_map( const char* szInput, map<string, string>& mssResult, 
		const char* szLineDelims = "\n", const char* szPairDelims = "=", bool bTrim = false );
	void map_implode( const map_ss& mssMap, string& sResult, char cLineDelim = '\n', char cPairDelim = '=' );

	char* find_first_not_of( const char* szInput, const char* szDelims = " \t" );

	size_t CharVectorToStringVector( const char* pszInput[], string_v& vsOutput );

	size_t GetSubstringEntries( const string& sInput, const char* szSubString, uint_v& viResult );
	string& ClearSpaces( string& sString, const char* szSpaces = " \t\r\n" );

	size_t ParseCommandLine( const char* szCommandLine, string_v& vsParams );
	string ParseCommandParams( const string& sCommandLine, string& sParams );

	inline bool IsSlash( const char chChar ) { return '\\' == chChar || '/' == chChar; }

	string LongToBitString( uint nValue );

	int CompareString(const char *szFirstValue, const char *szSecondValue, bool bCaseSensitive = false);
	bool CompareStr(const char *szFirstValue, const char *szSecondValue, bool bCaseSensitive = false);

	bool IsHexDigit(char chChar);
}

class StringFormater
{
public:
	template<class T> StringFormater& operator<< (const T& arg) 
	{
		m_ssStream << arg;
		return *this;
	}
	operator std::string() const 
	{
		return m_ssStream.str();
	}
protected:
	std::stringstream m_ssStream;
};


#endif // _STR_UTILS_
