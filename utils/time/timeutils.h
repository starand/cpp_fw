#ifndef __H_TIMEUTILS__
#define __H_TIMEUTILS__

#ifdef GetCurrentTime
#	undef GetCurrentTime
#endif

//////////////////////////////////////////////////
// namespace TimeUtils declaration

#define FILETIME2UINT64( _TIME_ ) ((((uint64)_TIME_.dwHighDateTime) << 32) | _TIME_.dwLowDateTime)
#define UINT642FILETIME( _VALUE_, _TIME_ ) _TIME_.dwHighDateTime = ( (_VALUE_ >> 32) & 0xFFFFFFFF ); _TIME_.dwLowDateTime = ( _VALUE_ & 0xFFFFFFFF );

namespace TimeUtils
{
	enum EDATATIMEFORMAT
	{
		EDTF_TIME = 0,
		EDTF_DATE_MDY,
		EDTF_DATE_YMD,
		EDTF_DATE_DMY,
		EDTF_DATETIME_MDY,
		EDTF_DATETIME_YMD,
		EDTF_DATETIME_DMY,
		EDTF_DATETIMEMS_MDY,
		EDTF_DATETIMEMS_YMD,
		EDTF_DATETIMEMS_DMY,
		EDTF_SECMS,

		EDTF_INVALID
	};

	bool GetCurrentTime( SYSTEMTIME& stLocalTime );
	bool GetCurrentTime( FILETIME& ftCurrentTime );
	bool GetCurrentTime( string& sTime, EDATATIMEFORMAT edtfForamt = EDTF_DATETIME_MDY );

	bool ConvertStringToSystemtime( const string& sTime, SYSTEMTIME& ftResult, EDATATIMEFORMAT edtfForamt = EDTF_DATETIME_MDY );
	bool ConvertSystemtimeToString( const SYSTEMTIME& ftTime, string& sResult, EDATATIMEFORMAT edtfForamt = EDTF_DATETIME_DMY );

	bool ConvertStringToFiletime( const string& sTime, FILETIME& ftResult, EDATATIMEFORMAT edtfForamt = EDTF_DATETIME_MDY );
	bool ConvertFiletimeToString( const FILETIME& ftTime, string& sResult, EDATATIMEFORMAT edtfForamt = EDTF_DATETIME_DMY );

	bool ConvertTimeFormat( string& sTime, EDATATIMEFORMAT edtfInputForamt, EDATATIMEFORMAT edtfOutputForamt );
	
	void GetTimeDiff( uint64 nTime, SYSTEMTIME& stTime );
	void GetTimeDiff( FILETIME tfStart, FILETIME ftEnd, SYSTEMTIME& stTime );
	string TimeDiffToString( const SYSTEMTIME& stTime );
	string GetTimeDiff( FILETIME tfStart, FILETIME ftEnd );

	FILETIME ConvertFileTime(const string &sFileTime);
	
};

bool operator<( const FILETIME& ftLeft, const FILETIME& ftRight );
bool operator==( const FILETIME& ftLeft, const FILETIME& ftRight );
bool operator>( const FILETIME& ftLeft, const FILETIME& ftRight );
FILETIME operator-( const FILETIME& ftLeft, const FILETIME& ftRight );

#endif // __H_TIMEUTILS__
