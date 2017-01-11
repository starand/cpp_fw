#include "stdafx.h"
#include "systeminfo.h"
#include <Lmcons.h>

namespace SystemInfo
{
	string GetUserName()
	{
		DWORD nLen = UNLEN;
		char szUserNameBuffer[ UNLEN + 1 ] = { 0 };
		return ( 0 != GetUserNameA(szUserNameBuffer, &nLen) ? szUserNameBuffer : "" ) ;
	}


};