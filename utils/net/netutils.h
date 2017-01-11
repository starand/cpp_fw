#ifndef __H_NETUTILS__
#define __H_NETUTILS__

#include <winsock2.h>
#include "types.h"

namespace NetUtils {

uint Ip2Uint( const char* szIpAddress );
const char* Uint2Ip( uint uIpAddress );

ushort CheckSum( ushort* ptr, size_t nBytes );

void ShowDataPart( const char* szData, ushort nLen, ushort nPartLen, char nShowChar, char nShowHex = -1 );

};

#endif // __H_NETUTILS__
