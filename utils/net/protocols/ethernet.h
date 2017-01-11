#ifndef __ETHER_HEADER__
#define __ETHER_HEADER__

#include "types.h"

#define	E_PUP			0x0200
#define	E_IP			0x0800
#define E_ARP			0x0806
#define E_RARP			0x8035
#define E_NS			0x0600
#define	E_SPRITE		0x0500
#define E_TRAIL			0x1000
#define	E_MOPDL			0x6001
#define	E_MOPRC			0x6002
#define	E_DN			0x6003
#define	E_LAT			0x6004
#define E_SCA			0x6007
#define	E_LANBRIDGE		0x8038
#define	E_DECDNS		0x803c
#define	E_DECDTS		0x803e
#define	E_VEXP			0x805b
#define	E_VPROD			0x805c
#define E_ATALK			0x809b
#define E_AARP			0x80f3
#define	E_8021Q			0x8100
#define E_IPX			0x8137
#define E_IPV6			0x86dd
#define	E_PPP			0x880b
#define	E_MPLS			0x8847
#define	E_MPLS_MULTI	0x8848
#define E_PPPOED		0x8863
#define E_PPPOES		0x8864
#define	E_LOOPBACK		0x9000
#define	E_VMAN	        0x9100
#define	E_ISO           0xfefe

class CEthernet
{
public:		
	void Create(char* pszEdest, char* pszEsrc, ushort nType = E_IP);
	
	const char*	GetDestAddress() const;
	const char*	GetSrcAddress() const;
	void SetFDest(char* pszDest);
	void SetFSrc(char* pszSrc);

	ushort Type() const;
	void SetType(ushort nType = 0x800);
	const char*	GetType() const;

	void SwapMac();

	uchar &operator[](int i);

	void Show( ulong ulParams );

protected:
	uchar	m_aDest[6];
	uchar	m_aSrc[6];
	ushort	m_nType;
};

ostream &operator<<(ostream &stream, const CEthernet& eth);

#endif // __ETHER_HEADER__
