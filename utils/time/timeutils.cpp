#include "stdafx.h"
#include "timeutils.h"

//////////////////////////////////////////////////
// namespace TimeUtils implementation

namespace TimeUtils
{
	bool GetCurrentTime( SYSTEMTIME& stLocalTime )
	{
		GetSystemTime( &stLocalTime );
		return true;
	}

	bool GetCurrentTime( FILETIME& ftCurrentTime )
	{
		SYSTEMTIME stSystemTime;
		return GetCurrentTime( stSystemTime ) && SystemTimeToFileTime( &stSystemTime, &ftCurrentTime );
	}

	bool GetCurrentTime( string& sTime, EDATATIMEFORMAT edtfForamt /*= EDTF_DATETIME_MDY*/ )
	{
		SYSTEMTIME stSystemTime;
		return GetCurrentTime( stSystemTime ) && ConvertSystemtimeToString( stSystemTime, sTime, edtfForamt );
	}

	bool ConvertStringToSystemtime( const string& sTime, SYSTEMTIME& stTime, EDATATIMEFORMAT edtfForamt /*= EDTF_DATETIME_MDY*/ )
	{
		uint nMillisec = 0, nMonth = 0;

		switch( edtfForamt )
		{
		case EDTF_TIME:
			sscanf_s( sTime.c_str(), "%02d:%02d:%02d", &stTime.wHour, &stTime.wMinute, &stTime.wSecond );
			break;

		case EDTF_DATE_MDY:
			sscanf_s( sTime.c_str(), "%02d/%02d/%04d", &nMonth, &stTime.wDay, &stTime.wYear );
			break;
		case EDTF_DATE_YMD:
			sscanf_s( sTime.c_str(), "%04d/%02d/%02d", &stTime.wYear, &nMonth, &stTime.wDay );
			break;
		case EDTF_DATE_DMY:
			sscanf_s( sTime.c_str(), "%02d/%02d/%04d", &stTime.wDay, &nMonth, &stTime.wYear );
			break;

		case EDTF_DATETIME_MDY:
			sscanf_s( sTime.c_str(), "%02d/%02d/%04d %02d:%02d:%02d", 
				&nMonth, &stTime.wDay, &stTime.wYear, &stTime.wHour, &stTime.wMinute, &stTime.wSecond );
			break;
		case EDTF_DATETIME_YMD:
			sscanf_s( sTime.c_str(), "%04d/%02d/%02d %02d:%02d:%02d", 
				&stTime.wYear, &nMonth, &stTime.wDay, &stTime.wHour, &stTime.wMinute, &stTime.wSecond );
			break;
		case EDTF_DATETIME_DMY:
			sscanf_s( sTime.c_str(), "%02d/%02d/%04d %02d:%02d:%02d", 
				&stTime.wDay, &nMonth, &stTime.wYear, &stTime.wHour, &stTime.wMinute, &stTime.wSecond );
			break;

		case EDTF_DATETIMEMS_MDY:
			sscanf_s( sTime.c_str(), "%02d/%02d/%044d %02d:%02d:%02d.%03d", 
				&nMonth, &stTime.wDay, &stTime.wYear, &stTime.wHour, &stTime.wMinute, &stTime.wSecond, &nMillisec );
			break;
		case EDTF_DATETIMEMS_YMD:
			sscanf_s( sTime.c_str(), "%04d/%02d/%02d %02d:%02d:%02d.%03d", 
				&stTime.wYear, &nMonth, &stTime.wDay, &stTime.wHour, &stTime.wMinute, &stTime.wSecond, &nMillisec );
			break;
		case EDTF_DATETIMEMS_DMY:
			sscanf_s( sTime.c_str(), "%02d/%02d/%04d %02d:%02d:%02d.%03d", 
				&stTime.wDay, &nMonth, &stTime.wYear, &stTime.wHour, &stTime.wMinute, &stTime.wSecond, &nMillisec );
			break;

		case EDTF_SECMS:
			sscanf_s( sTime.c_str(), "%02d.%03d", &stTime.wSecond, &nMillisec );
			break;
		}

		stTime.wMonth = nMonth;
		stTime.wMilliseconds = nMillisec;
		if( stTime.wYear < 100 ) stTime.wYear += 2000;

		return true;
	}

	bool ConvertSystemtimeToString( const SYSTEMTIME& stTime, string& sResult, EDATATIMEFORMAT edtfForamt /*= EDTF_DATETIME_DMY*/ )
	{
		switch( edtfForamt )
		{
		case EDTF_TIME: {
			static const char szTimeForamt[] = "%02d:%02d:%02d";
			FormatString( sResult, szTimeForamt, stTime.wHour, stTime.wMinute, stTime.wSecond );
			break; }

		case EDTF_DATE_MDY: {
			static const char szTimeForamt[] = "%02d/%02d/%04d";
			FormatString( sResult, szTimeForamt, stTime.wMonth, stTime.wDay, stTime.wYear );
			break; }
		case EDTF_DATE_YMD: {
			static const char szTimeForamt[] = "%04d/%02d/%02d";
			FormatString( sResult, szTimeForamt, stTime.wYear, stTime.wMonth, stTime.wDay );
			break; }
		case EDTF_DATE_DMY: {
			static const char szTimeForamt[] = "%02d/%02d/%04d";
			FormatString( sResult, szTimeForamt, stTime.wDay, stTime.wMonth, stTime.wYear );
			break; }

		case EDTF_DATETIME_MDY: {
			static const char szTimeForamt[] = "%02d/%02d/%04d %02d:%02d:%02d";
			FormatString( sResult, szTimeForamt, stTime.wMonth, stTime.wDay, stTime.wYear, stTime.wHour, stTime.wMinute, stTime.wSecond );
			break; }
		case EDTF_DATETIME_YMD: {
			static const char szTimeForamt[] = "%04d/%02d/%02d %02d:%02d:%02d";
			FormatString( sResult, szTimeForamt, stTime.wYear, stTime.wMonth, stTime.wDay, stTime.wHour, stTime.wMinute, stTime.wSecond );
			break; }
		case EDTF_DATETIME_DMY: {
			static const char szTimeForamt[] = "%02d/%02d/%04d %02d:%02d:%02d";
			FormatString( sResult, szTimeForamt, stTime.wDay, stTime.wMonth, stTime.wYear, stTime.wHour, stTime.wMinute, stTime.wSecond );
			break; }

		case EDTF_DATETIMEMS_MDY: {
			static const char szTimeForamt[] = "%02d/%02d/%04d %02d:%02d:%02d.%03d";
			FormatString( sResult, szTimeForamt, stTime.wMonth, stTime.wDay, stTime.wYear, stTime.wHour, stTime.wMinute, stTime.wSecond, stTime.wMilliseconds );
			break; }
		case EDTF_DATETIMEMS_YMD: {
			static const char szTimeForamt[] = "%04d/%02d/%02d %02d:%02d:%02d.%03d";
			FormatString( sResult, szTimeForamt, stTime.wYear, stTime.wMonth, stTime.wDay, stTime.wHour, stTime.wMinute, stTime.wSecond, stTime.wMilliseconds );
			break; }
		case EDTF_DATETIMEMS_DMY: {
			static const char szTimeForamt[] = "%02d/%02d/%04d %02d:%02d:%02d.%03d";
			FormatString( sResult, szTimeForamt, stTime.wDay, stTime.wMonth, stTime.wYear, stTime.wHour, stTime.wMinute, stTime.wSecond, stTime.wMilliseconds );
			break; }

		case EDTF_SECMS: {
			static const char szTimeForamt[] = "%02d.%03d";
			FormatString( sResult, szTimeForamt, stTime.wSecond, stTime.wMilliseconds );
			break; }
		}

		return true;
	}

	bool ConvertStringToFiletime( const string& sTime, FILETIME& ftResult, EDATATIMEFORMAT edtfForamt /*= EDTF_DATETIME_MDY*/ )
	{ 
		SYSTEMTIME stTime = { 0 };
		return ConvertStringToSystemtime( sTime, stTime, edtfForamt ) && SystemTimeToFileTime( &stTime, &ftResult );
	}

	bool ConvertFiletimeToString( const FILETIME& ftTime, string& sResult, EDATATIMEFORMAT edtfForamt /*= EDTF_DATETIME_DMY*/ )
	{
		sResult.erase();
		SYSTEMTIME stUTC, stLocal;
		return FileTimeToSystemTime( &ftTime, &stUTC )
			&& SystemTimeToTzSpecificLocalTime( NULL, &stUTC, &stLocal ) 
			&& ConvertSystemtimeToString( stLocal, sResult, edtfForamt );
	}

	bool ConvertTimeFormat( string& sTime, EDATATIMEFORMAT edtfInputForamt, EDATATIMEFORMAT edtfOutputForamt )
	{
		SYSTEMTIME stTime;
		return ConvertStringToSystemtime( sTime, stTime, edtfInputForamt ) && ConvertSystemtimeToString( stTime, sTime, edtfOutputForamt );
	}

	void GetTimeDiff( uint64 nDiff, SYSTEMTIME& stTime )
	{
		nDiff /= 10000; // convert to milliseconds

		stTime.wMilliseconds = nDiff % 1000; nDiff /= 1000;
		stTime.wSecond = nDiff % 60; nDiff /= 60;
		stTime.wMinute = nDiff % 60; nDiff /= 60;
		stTime.wHour = nDiff % 24; nDiff /= 24;
		stTime.wDay = (WORD)nDiff;
	}

	void GetTimeDiff( FILETIME tfStart, FILETIME ftEnd, SYSTEMTIME& stTime )
	{
		uint64 nDiff = _abs64( FILETIME2UINT64(ftEnd) - FILETIME2UINT64(tfStart) );
		GetTimeDiff( nDiff, stTime );
	}
	
	string TimeDiffToString( const SYSTEMTIME& stTime )
	{
		static const char szFormat[] = "%02d:%02d:%02d.%03d";
		static const char szDaysFormat[] = "%u %02d:%02d:%02d.%03d";
		
		string sResult;
		if( stTime.wDay ) FormatString( sResult, szDaysFormat, stTime.wDay, stTime.wHour, stTime.wMinute, stTime.wSecond, stTime.wMilliseconds );
		else FormatString( sResult, szFormat, stTime.wHour, stTime.wMinute, stTime.wSecond, stTime.wMilliseconds );
		return sResult;
	}

	string GetTimeDiff( FILETIME tfStart, FILETIME ftEnd )
	{
		SYSTEMTIME stTime;
		GetTimeDiff( tfStart, ftEnd, stTime );
		return TimeDiffToString( stTime );
	}


	FILETIME ConvertFileTime(const string &sFileTime)
	{
		uint64 uiFileTime = StrUtils::StrToUInt64(sFileTime);

		FILETIME ftResultTime;
		UINT642FILETIME(uiFileTime, ftResultTime);

		return ftResultTime;
	}

};

bool operator<( const FILETIME& ftLeft, const FILETIME& ftRight ) { return ( FILETIME2UINT64(ftLeft) < FILETIME2UINT64(ftRight) ); }
bool operator==( const FILETIME& ftLeft, const FILETIME& ftRight ) { return ( FILETIME2UINT64(ftLeft) == FILETIME2UINT64(ftRight) ); }
bool operator>( const FILETIME& ftLeft, const FILETIME& ftRight ) { return ( FILETIME2UINT64(ftLeft) > FILETIME2UINT64(ftRight) ); }

FILETIME operator-( const FILETIME& ftLeft, const FILETIME& ftRight )
{
	uint64 nTimeOne = FILETIME2UINT64( ftLeft );
	uint64 nTimeTwo = FILETIME2UINT64( ftRight );
	uint64 nResult = ( nTimeOne > nTimeTwo ? nTimeOne - nTimeTwo : nTimeTwo - nTimeOne );

	FILETIME ftResult;
	UINT642FILETIME( nResult, ftResult );
	return ftResult;
}