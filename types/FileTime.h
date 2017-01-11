#ifndef __FILETIME_H_INCLUDED
#define __FILETIME_H_INCLUDED

#include <Winbase.h>

//////////////////////////////////////////////////
// CFileTime declaration

struct CFileTime : public FILETIME
{
	CFileTime() { dwHighDateTime = dwLowDateTime = 0; }
	CFileTime(const FILETIME &ftFileTime) { dwHighDateTime = ftFileTime.dwHighDateTime; dwLowDateTime = ftFileTime.dwLowDateTime; }

	~CFileTime() { }

	operator FILETIME() { return static_cast<FILETIME>(*this); }
	bool operator==(const CFileTime &ftAnotherTime) const { return dwHighDateTime == ftAnotherTime.dwHighDateTime && dwLowDateTime == ftAnotherTime.dwLowDateTime; }
};

#endif // __FILETIME_H_INCLUDED
