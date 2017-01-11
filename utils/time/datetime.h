#ifndef __H_TIME__
#define __H_TIME__

#include <Windows.h>

enum EDTFORMAT
{
	EDTF_TIME = 0,
	EDTF_DATE,
	EDTF_DATETIME,
	EDTF_SECMS,

	EDTF_INVALID
};

struct CDataTimeStorage
{
	CDataTimeStorage();
	CDataTimeStorage( const SYSTEMTIME& stTime );

	friend class CDateTime;

private:
	uint64 GetInMilliseconds() const;
	void SetInMilliseconds( uint64 uiMillisecs );

	CDataTimeStorage operator+( const CDataTimeStorage& opRight ) const;
	CDataTimeStorage operator-( const CDataTimeStorage& opRight ) const;

protected:
	CDataTimeStorage& Normalize();

public:
	short wYear;
	char  ucMonth;
	char  ucDay;
	char  ucHour;
	char  ucMinute;
	short wSecond;
	short wMilliseconds;
};

class CDateTime
{
public:
	CDateTime();
	CDateTime( uint64 nMillisecs );
	CDateTime( const char* szTime );

	void ScanFromString( const char* szTime, EDTFORMAT edtfFormat = EDTF_DATETIME );
	const char* GetTimeStr( EDTFORMAT edtfFormat = EDTF_DATETIME ) const;

	static const char* GetTimeStr( const CDataTimeStorage& stTime, EDTFORMAT edtfFormat = EDTF_DATETIME );
	static const char* Now( EDTFORMAT edtfFormat = EDTF_DATETIME );

	static FILETIME NowFT(bool bUTC = false);

	uint64 ToMillisecs() const;

	CDataTimeStorage& GetStorage();
	const CDataTimeStorage& GetStorage() const;

	bool operator<( const CDateTime& dtRight ) const;
	CDateTime operator-( const CDateTime& dtRight );
	CDateTime operator+( const CDateTime& dtRight );
	const CDateTime operator+( const CDateTime& dtRight ) const;

	CDateTime& PlusDay();
	bool GetIsNULL() const;

protected:
	bool IsYearIntercalary() const;
	uchar GetDaysCurrMonth() const;

	static EDTFORMAT DetermineFormat( const char* szTimeString );
	void SetNow();

private:

	CDataTimeStorage m_dtsTime;
};

#define TIME_START() double __tm_start__ = atof( CDateTime::Now(EDTF_SECMS) )
#define TIME_RESTART() __tm_start__ = atof( CDateTime::Now(EDTF_SECMS) )
#define TIME_DIFF() ( atof(CDateTime::Now(EDTF_SECMS)) - __tm_start__ )
#define TIME_DIFF_MSG( x ) \
	cout << " [ " << ( atof(CDateTime::Now(EDTF_SECMS)) - __tm_start__ ) << " ] " << (x) << endl; \
	TIME_RESTART(); 

#endif // __H_TIME__
