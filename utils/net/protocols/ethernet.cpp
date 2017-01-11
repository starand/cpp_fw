#include "stdafx.h"
#include "ethernet.h"
#include "strutils.h"
#include "stdutils.h"

using namespace StdUtils;

#include <WinSock2.h>
#pragma comment( lib, "ws2_32.lib" )

uchar& CEthernet::operator[](int nPos)
{
	return *((uchar *)this + nPos);
}

void CEthernet::Create(char* pszEdest, char* pszEsrc, ushort nType /* = E_IP */)
{
	SetFDest( pszEdest );
	SetFSrc( pszEsrc );
	SetType( nType );
}

const char* CEthernet::GetDestAddress() const
{
	return StrUtils::MacAddr(m_aDest);
}

const char* CEthernet::GetSrcAddress() const
{
	return StrUtils::MacAddr(m_aSrc);
}

ushort CEthernet::Type() const
{
	return ntohs(m_nType);
}

void CEthernet::SetType(ushort nType /* = 0x800 */)
{
	m_nType = htons(nType);
}

void CEthernet::SetFDest(char* pszDest)
{
	int a, b, c, x, y, z;
	sscanf_s(pszDest, "%2X:%2X:%2X:%2X:%2X:%2X", &a, &b, &c, &x, &y, &z);
	m_aDest[0] = a; m_aDest[1] = b; m_aDest[2] = c; m_aDest[3] = x; m_aDest[4] = y; m_aDest[5] = z;
}

void CEthernet::SetFSrc(char* pszSrc)
{
	int a, b, c, x, y, z;
	sscanf_s(pszSrc, "%2X:%2X:%2X:%2X:%2X:%2X", &a, &b, &c, &x, &y, &z);
	m_aSrc[0] = a; m_aSrc[1] = b; m_aSrc[2] = c; m_aSrc[3] = x; m_aSrc[4] = y; m_aSrc[5] = z;
}

const char* CEthernet::GetType() const
{
	static string sResult;

	switch( Type() )
	{
	case  E_PUP :	sResult.assign(" PUP "); break;
	case  E_IP :	sResult.assign(" IP "); break;
	case  E_ARP :	sResult.assign(" ARP "); break;
	case  E_RARP :	sResult.assign(" RARP "); break;
	case  E_NS :	sResult.assign(" NS "); break;
	case  E_SPRITE: sResult.assign(" SPRITE "); break;
	case  E_TRAIL :	sResult.assign(" TRAIL "); break;
	case  E_MOPDL :	sResult.assign(" MOPDL "); break;
	case  E_MOPRC :	sResult.assign(" MOPRC "); break;
	case  E_DN :	sResult.assign(" DN "); break;
	case  E_LAT :	sResult.assign(" LAT "); break;
	case  E_SCA :	sResult.assign(" SCA "); break;
	case  E_LANBRIDGE : sResult.assign(" LANBRIDGE "); break;
	case  E_DECDNS:	sResult.assign(" DECDNS "); break;
	case  E_DECDTS:	sResult.assign(" DECDTS "); break; 
	case  E_VEXP :	sResult.assign(" VEXP "); break;
	case  E_VPROD :	sResult.assign(" VPROD "); break;
	case  E_ATALK :	sResult.assign(" ATALK "); break;
	case  E_AARP :	sResult.assign(" AARP "); break;
	case  E_8021Q : sResult.assign(" 8021Q "); break;
	case  E_IPX :	sResult.assign(" IPX "); break;
	case  E_IPV6 :	sResult.assign(" IPV6 "); break;
	case  E_PPP :	sResult.assign(" PPP "); break;
	case  E_MPLS :	sResult.assign(" MPLS "); break;
	case  E_MPLS_MULTI : sResult.assign(" MPLS_MULTI "); break;
	case  E_PPPOED: sResult.assign(" PPPOED "); break;
	case  E_PPPOES: sResult.assign(" PPPOES "); break;
	case  E_LOOPBACK : sResult.assign(" LOOPBACK "); break;
	case  E_VMAN :	sResult.assign(" VMAN "); break;
	case  E_ISO :	sResult.assign(" ISO "); break;
	default: sResult.assign( StrUtils::FormatString("%x",Type())); break;
	};

	return sResult.c_str();
}

void CEthernet::SwapMac()
{
	uchar tmp[6];
	memcpy(tmp, m_aDest, 6);
	memcpy(m_aDest, m_aSrc, 6);
	memcpy(m_aSrc, tmp, 6);
}

ostream &operator<<(ostream &stream, const CEthernet& eth)
{
	stream << "Eth " << eth.GetSrcAddress() << " > " << eth.GetDestAddress() << " " << eth.GetType();
	return stream;
}

void CEthernet::Show( ulong ulParams )
{
	cout << save << purple << "Eth " << aqua << GetSrcAddress() << restore << " > ";
	cout << green <<  GetDestAddress() << " " << red << GetType() << endl << restore;
}
