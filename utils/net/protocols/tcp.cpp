#include "stdafx.h"
#include "tcp.h"
#include "ip.h"
#include "formatparser.h"
#include "stdutils.h"
#include "netutils.h"

using namespace StdUtils;
using namespace NetUtils;

/////////////////////////////////////////////////////////
// CTcp implementation

ushort CTcp::GetDataLen( CIp* pIpHeader ) const 
{
	return pIpHeader->GetDataLen() - GetHeaderLen(); 
}

ushort CTcp::GetHeaderLen() const
{
	return ( offset >> 2 );
}

void CTcp::SetOffset( ushort nOffset )
{
	offset = nOffset << 2;
}

uchar& CTcp::operator[]( size_t i )
{
	return *((uchar *)this + i);
}

ushort CTcp::GetSrcPort() const
{
	return ntohs( sport );
}

ushort CTcp::GetDestPort() const
{
	return ntohs( dport );
}

void CTcp::SetSrcPort( ushort nPort )
{
	sport = htons(nPort);
}

void CTcp::SetDesrPort( ushort nPort )
{
	dport = htons( nPort );
}

uint CTcp::GetSN() const
{
	return ntohl( sn );
}

uint CTcp::GetAN() const
{
	return ntohl( an );
}

ushort CTcp::GetOffset() const
{
	return offset >> 2;
}

void CTcp::SetSN( uint uiSN )
{
	sn = htonl( uiSN );
}

void CTcp::SetAN( uint uiAN )
{
	an = htonl( uiAN );
}

ushort CTcp::GetWindow() const
{
	return ntohs( window );
}

void CTcp::SetWindow( ushort usWindow )
{ 
	window = htons( usWindow );
}

ushort CTcp::GetUrgPoint() const
{
	return ntohs( urg_point );
}

void CTcp::Show( CIp* pIpHeader, ulong ulParams ) const
{
	if( pIpHeader ) 
	{
		if( ulParams & FOP_ANALYSED )
		{
			cout << save << purple << pIpHeader->GetProtocol() << " " << aqua << pIpHeader->GetSrcAddress();
			cout << ":" << lightaqua << GetSrcPort() << restore << " > " << green << pIpHeader->GetDestAddress();
			cout << ":" << lightgreen << GetDestPort() << red << " " << GetFlagsString() << restore;
			
			if( ulParams & FOP_ANALYSEDFULL )
			{
				cout << " " << gray << GetSN() << " " << yellow << GetAN();
				cout << " " << purple << GetHeaderLen() << "-" << GetDataLen(pIpHeader) << " " << red << GetCS() << restore;
			}

			cout << endl;
		}
		
		bool bShowChars = (ulParams & FOP_SHOWINCHARS) == FOP_SHOWINCHARS;
		bool bShowHex = (ulParams & FOP_SHOWINHEX) == FOP_SHOWINHEX;

		bool bShowOnlyData = ( ulParams & FOP_SHOwDATA ) == FOP_SHOwDATA;
		
		if( (bShowChars || bShowHex) && bShowOnlyData )
		{
			const char* szData = (const char*)this->GetData();
			ushort nDataLen = this->GetDataLen(pIpHeader);

			ushort nPartLen = min( 0xFFF, nDataLen );
			if( bShowHex ) nPartLen = 32;
			if( nDataLen > 1 ) 
			{
				ShowDataPart( szData, nDataLen, nPartLen, 
					bShowChars ? WHITE : -1, bShowHex ? GRAY : -1 );

				cout << endl; // enf of packet
			}
		}
	}
}

uchar* CTcp::GetData() const
{
	return (uchar *)((char *)this + GetOffset());
}

void CTcp::SwapPort()
{
	ushort tmp = sport; sport = dport; dport = tmp;
}

ushort CTcp::GetCS() const
{
	return ntohs( cs );
}

const char* CTcp::GetFlagsString() const
{
	static const char szBufferSize = 14;
	static char szFlagsBuffer[ szBufferSize ] = { 0 };

	sprintf_s( szFlagsBuffer, szBufferSize-1, "%s%s%s%s%s%s",
		(fl&T_FIN ? "F " : ""), (fl&T_SYN ? "S " : ""), (fl&T_RST ? "R " : ""), 
		(fl&T_PUSH ? "P " : ""),(fl&T_ACK ? "A " : ""),(fl&T_URG ? "U " : "") );

	return (const char*)szFlagsBuffer;
}
