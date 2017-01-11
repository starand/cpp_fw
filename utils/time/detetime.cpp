#include "stdafx.h"
#include "datetime.h"
#include "types.h"
#include "macroes.h"

const char szTimeErrorString[] = "Time error";

uchar aucMonthDays[] = { 0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31 };
uchar aucMonthDaysI[] = { 0, 31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31 };

enum EMONTHS { EM_JAN = 1, EM_FEB, EM_MAR, EM_APR, EM_MAY, EM_JUN, EM_JUL, EM_AUG, EM_SEP, EM_OCT, EM_NOV, EM_DEC };

///////////////////////////////////////////////////////////
// CDateTime implementation

CDateTime::CDateTime() : m_dtsTime()
{
	SetNow();
}

CDateTime::CDateTime( uint64 nMillisecs ) : m_dtsTime()
{
	GetStorage().SetInMilliseconds( nMillisecs );
}

CDateTime::CDateTime( const char* szTime ) : m_dtsTime()
{
	SetNow();
	EDTFORMAT edtfFormat = DetermineFormat( szTime );
	ScanFromString( szTime, edtfFormat );
}

void CDateTime::SetNow()
{
	SYSTEMTIME stTime;
	GetLocalTime( &stTime );
	m_dtsTime = stTime;
}

void CDateTime::ScanFromString( const char* szTime, EDTFORMAT edtfFormat /*= EDTF_DATETIME*/ )
{
	WORD wSeconds = 0, wMilliseconds = 0, wHours = 0, wMinutes = 0, wDays = 0, wMonth = 0;
	switch( edtfFormat )
	{
	case EDTF_TIME: // "23:08:26"
		sscanf_s( szTime, "%hu:%hu:%hu", &wHours, &wMinutes, &wSeconds );
		m_dtsTime.ucHour = (uchar)wHours;
		m_dtsTime.ucMinute = (uchar)wMinutes;
		m_dtsTime.wSecond = wSeconds;
		m_dtsTime.wMilliseconds = 0;
		break;
	case EDTF_DATE: // "23:08:26 23/03/2013"
		sscanf_s( szTime, "%hu/%hu/%hu", &wDays, &wMonth, &m_dtsTime.wYear );
		m_dtsTime.ucDay = (uchar)wDays;
		m_dtsTime.ucMonth = (uchar)wMonth;
		break;
	case EDTF_DATETIME: // "23:08:26 23/03/2013"
		sscanf_s( szTime, "%hu/%hu/%hu %hu:%hu:%hu ", &wDays, &wMonth, &m_dtsTime.wYear, &wHours, &wMinutes, &wSeconds );
		m_dtsTime.ucHour = (uchar)wHours;
		m_dtsTime.ucMinute = (uchar)wMinutes;
		m_dtsTime.ucDay = (uchar)wDays;
		m_dtsTime.ucMonth = (uchar)wMonth;
		m_dtsTime.wSecond = wSeconds;
		m_dtsTime.wMilliseconds = 0;
		break;
	case EDTF_SECMS:
		sscanf_s( szTime, "%hu.%hu", &wSeconds, &wMilliseconds );
		m_dtsTime.wSecond = wSeconds; 
		m_dtsTime.wMilliseconds = wMilliseconds;
		break;
	}
}

const char* CDateTime::GetTimeStr( EDTFORMAT edtfFormat ) const
{
	return GetTimeStr( m_dtsTime, edtfFormat );
}

const char* CDateTime::GetTimeStr( const CDataTimeStorage& stTime, EDTFORMAT edtfFormat )
{
	static const size_t nBufferSize = 64;
	static char szTimeBuffer[ nBufferSize ];

	switch( edtfFormat )
	{
	case EDTF_TIME:
		sprintf_s( (char*)szTimeBuffer, nBufferSize - 1, "%2.2i:%2.2i:%2.2i", stTime.ucHour, stTime.ucMinute, stTime.wSecond );
		break;
	case EDTF_DATE:
		sprintf_s( (char*)szTimeBuffer, nBufferSize - 1, "%2.2i/%2.2i/%i", stTime.ucDay, stTime.ucMonth, stTime.wYear );
		break;
	case EDTF_SECMS:
		sprintf_s( (char*)szTimeBuffer, nBufferSize - 1, "%i.%3.3i", stTime.wSecond, stTime.wMilliseconds );
		break;
	default:
		sprintf_s( (char*)szTimeBuffer, nBufferSize - 1, "%2.2i:%2.2i:%2.2i %2.2i/%2.2i/%i", 
			stTime.ucHour, stTime.ucMinute, stTime.wSecond, stTime.ucDay, stTime.ucMonth, stTime.wYear );
	};

	return (const char*)szTimeBuffer;
}

const char* CDateTime::Now( EDTFORMAT edtfFormat )
{
	SYSTEMTIME stTime;
	GetSystemTime( &stTime );

	return GetTimeStr( stTime, edtfFormat );
}


/*static */
FILETIME CDateTime::NowFT(bool bUTC /*= false*/)
{
	SYSTEMTIME stTime;

	if (bUTC)
	{
		GetSystemTime(&stTime);
	}
	else
	{
		GetLocalTime(&stTime);
	}

	FILETIME ftResult;
	SystemTimeToFileTime(&stTime, &ftResult);

	return ftResult;
}


CDataTimeStorage& CDateTime::GetStorage()
{
	return m_dtsTime;
}

const CDataTimeStorage& CDateTime::GetStorage() const
{
	return m_dtsTime;
}

CDateTime CDateTime::operator-( const CDateTime& dtRight )
{
	CDateTime dtResult;
	dtResult.GetStorage() = GetStorage().operator-( dtRight.GetStorage() );
	return dtResult;
}

CDateTime CDateTime::operator+( const CDateTime& dtRight )
{
	CDateTime dtResult;
	dtResult.GetStorage() = GetStorage().operator+( dtRight.GetStorage() );
	return dtResult;
}

const CDateTime CDateTime::operator+( const CDateTime& dtRight ) const
{
	CDateTime dtResult;
	dtResult.GetStorage() = GetStorage().operator+( dtRight.GetStorage() );
	return dtResult;
}

bool CDateTime::operator<( const CDateTime& dtRight ) const
{
	return m_dtsTime.GetInMilliseconds() < dtRight.m_dtsTime.GetInMilliseconds();
}

uint64 CDateTime::ToMillisecs() const
{
	return m_dtsTime.GetInMilliseconds();
}

bool CDateTime::IsYearIntercalary() const
{
	return ( (m_dtsTime.wYear % 4 == 0 && m_dtsTime.wYear % 100 != 0) || m_dtsTime.wYear % 400 == 0 );
}

uchar CDateTime::GetDaysCurrMonth() const
{
	return ( IsYearIntercalary() ? aucMonthDaysI[ m_dtsTime.ucMonth ] : aucMonthDays[ m_dtsTime.ucMonth ] );
}

CDateTime& CDateTime::PlusDay()
{
	if( m_dtsTime.ucDay < GetDaysCurrMonth() ) ++m_dtsTime.ucDay;
	else  
	{
		if( m_dtsTime.ucMonth == EM_DEC )
		{
			++m_dtsTime.wYear;
			m_dtsTime.ucMonth = m_dtsTime.ucDay = 1;
		}
		else
		{
			++m_dtsTime.ucMonth;
			m_dtsTime.ucDay = 1;
		}
	}

	return *this;
}

bool CDateTime::GetIsNULL() const
{
	return ( 0 == GetStorage().GetInMilliseconds() );
}

EDTFORMAT CDateTime::DetermineFormat( const char* szTimeString )
{
	EDTFORMAT edtResult = EDTF_INVALID;

	if( strlen(szTimeString) >= 8 && szTimeString[2] == ':' && szTimeString[5] == ':' )
	{
		if( strlen(szTimeString) == 19 && szTimeString[11] == '/' && szTimeString[14] == '/' ) edtResult = EDTF_DATETIME;
		else edtResult = EDTF_TIME;
	}
	else if( strlen(szTimeString) == 10 && szTimeString[2] == '/' && szTimeString[5] == '/' ) edtResult = EDTF_DATE;
	else if( strchr(szTimeString, '.') ) edtResult = EDTF_SECMS;

	return edtResult;
}

///////////////////////////////////////////////////////////
// CDataTimeStorage implementation

CDataTimeStorage::CDataTimeStorage()  : wYear(), ucMonth(), ucDay(), ucHour(), ucMinute(), wSecond(), wMilliseconds() { }

#pragma warning( disable : 4244 )
CDataTimeStorage::CDataTimeStorage( const SYSTEMTIME& stTime ) : wYear(stTime.wYear), ucMonth(stTime.wMonth), ucDay(stTime.wDay), 
	ucHour(stTime.wHour), ucMinute(stTime.wMinute), wSecond(stTime.wSecond), wMilliseconds(stTime.wMilliseconds)
{ 
}
#pragma warning( default : 4244 )

uint64 CDataTimeStorage::GetInMilliseconds() const 
{
	uint64 nResult = ((wYear * 12) + ucMonth) * 30 + ucDay;
	return (((((nResult * 24 + ucHour) * 60) + ucMinute) * 60) + wSecond) * 1000 + wMilliseconds;
}

void CDataTimeStorage::SetInMilliseconds( uint64 uiMillisecs ) 
{
	wMilliseconds	= (short)uiMillisecs % 1000;	uiMillisecs /= 1000;
	wSecond			= (short)uiMillisecs % 60;		uiMillisecs /= 60;
	ucMinute		= (char) uiMillisecs % 60;		uiMillisecs /= 60;
	ucHour			= (char) uiMillisecs % 24;		uiMillisecs /= 24;
	ucDay			= (char) uiMillisecs % 30;		uiMillisecs /= 30;
	ucMonth			= (char) uiMillisecs % 12;		uiMillisecs /= 12;
	wYear			= (short)uiMillisecs;
}

#define ADD( x, y, z ) x.z += y.z;
#define MINUS( x, y, z ) x.z -= y.z;

CDataTimeStorage CDataTimeStorage::operator+( const CDataTimeStorage& opRight ) const
{
	CDataTimeStorage dtsResult( *this );

	ADD( dtsResult, opRight, wMilliseconds );
	ADD( dtsResult, opRight, wSecond );
	ADD( dtsResult, opRight, ucMinute );
	ADD( dtsResult, opRight, ucHour );
	ADD( dtsResult, opRight, ucDay );
	ADD( dtsResult, opRight, ucMonth );
	ADD( dtsResult, opRight, wYear );

	return dtsResult.Normalize();
}

CDataTimeStorage CDataTimeStorage::operator-( const CDataTimeStorage& opRight ) const
{
	CDataTimeStorage dtsResult( *this );

	MINUS( dtsResult, opRight, wMilliseconds );
	MINUS( dtsResult, opRight, wSecond );
	MINUS( dtsResult, opRight, ucMinute );
	MINUS( dtsResult, opRight, ucHour );
	MINUS( dtsResult, opRight, ucDay );
	MINUS( dtsResult, opRight, ucMonth );
	MINUS( dtsResult, opRight, wYear );

	return dtsResult.Normalize();
}

#define MILLISEC	1000
#define SECONDS		60
#define MINUTES		60
#define HOURS		24
#define DAYS		30
#define MONTHS		12

CDataTimeStorage& CDataTimeStorage::Normalize()
{
	// milliseconds
	while( wMilliseconds >= MILLISEC ) { wMilliseconds -= MILLISEC; ++wSecond; }
	while( wMilliseconds < 0 ) { wMilliseconds += MILLISEC; --wSecond; }
	// seconds
	while( wSecond >= SECONDS ) { wSecond -= SECONDS; ++ucMinute; }
	while( wSecond < 0 ) { wSecond += SECONDS; --ucMinute; }
	// minutes
	while( ucMinute >= MINUTES ) { ucMinute -= MINUTES; ++ucHour; }
	while( ucMinute < 0 ) { ucMinute += MINUTES; --ucHour; }
	// hours
	while( ucHour >= HOURS ) { ucHour -= HOURS; ++ucDay; }
	while( ucHour < 0 ) { ucHour += HOURS; --ucDay; }
	// days
	while( ucDay >= DAYS ) { ucDay -= DAYS; ++ucMonth; }
	while( ucDay < 0 ) { ucDay += DAYS; --ucMonth; }
	// month
	while( ucMonth >= MONTHS ) { ucMonth -= MONTHS; ++wYear; }
	while( ucMonth < 0 ) { ucMonth += MONTHS; --wYear; }

	return *this;
}
