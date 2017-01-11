#include "StdAfx.h"

#include "netutils.h"

#include <iomanip>


#ifdef __H_STDUTILS__
#	define SET_TEXT_ATTRS(__VAL__) StdUtils::SetTextAttribs( __VAL__ );
#	define GET_TEXT_ATTRS() StdUtils::GetTextAttribs()
#else
#	define SET_TEXT_ATTRS(__VAL__)
#	define GET_TEXT_ATTRS() 0
#endif

namespace NetUtils {

uint Ip2Uint( const char* szIpAddress )
{
	return ntohl( inet_addr( szIpAddress ) );
}

const char* Uint2Ip( uint uIpAddress )
{
	in_addr tmpInAddr;
	tmpInAddr.S_un.S_addr = uIpAddress;
	return inet_ntoa( tmpInAddr );		
}

ushort CheckSum( ushort* ptr, size_t nBytes )
{
	register long nSum = 0;
	ushort oddbyte;
	register ushort answer;

	while( nBytes > 1 )  
	{
		nSum += *ptr++;
		nBytes -= 2;
	}

	if( nBytes == 1 ) 
	{
		oddbyte = 0;
		*((u_char *) &oddbyte) = *(u_char *)ptr; 
		nSum += oddbyte;
	}

	nSum  = (nSum >> 16) + (nSum & 0xffff);
	nSum += (nSum >> 16);
	answer = (ushort)~nSum;

	return(answer);
}

void ShowDataPart( const char* szInputData, ushort nDataLen, ushort nPartLen, char nShowChar, char nShowHex )
{
	if( nShowChar == -1 && nShowHex == - 1 ) return;

	ushort usTextAttribs = GET_TEXT_ATTRS();
	for( ushort idx = 0; idx < nDataLen; idx += nPartLen )
	{
		ushort nLen = min( nDataLen - idx, nPartLen );
		const char* szData = szInputData + idx;

		ushort nOut = nPartLen;
		if( nShowChar >= 0 )
		{
			SET_TEXT_ATTRS(nShowChar);
			for( ushort i = 0; i < nLen; ++i, --nOut )
			{
				if( nShowHex >= 0 && !isprint((uchar)szData[i]) ) 
				{
					cout << ".";
					continue;
				}
				if( szData[i] == 7 || szData[i] == 8 || szData[i] == 13 ) continue;
				cout << szData[i];
			}
		}

		if( nShowHex >= 0 )
		{
			SET_TEXT_ATTRS(nShowHex);
			if( nShowChar >= 0 ) 
			{
				for( ; nOut; --nOut ) cout << " ";
				cout << "\t";
			}

			for( ushort i = 0; i < nLen; ++i )
			{
				cout << (hex) << setw(2) << setfill('0') << (int)(szData[i] & 0xFF) << " ";
			}
		}
	}

	SET_TEXT_ATTRS(usTextAttribs);
	cout << dec;
}

};
