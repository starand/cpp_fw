#ifndef __CARP_HEADER__
#define __CARP_HEADER__

#include "types.h"

class CArp
{
public:
	const char* GetEtherDest() const;
	const char* GetEtherSrc() const;
	const char* GetIpSrc() const;
	const char* GetIpDest() const;

	void SetEtherDest(char* pszDest);
	void SetEtherSrc(char* pszSrc);
	void SetIpDest(char* pszDest);
	void SetIpSrc(char* pszSrc);

	string GetHardware();

	ushort OpCode() const;
	string GetOpcode() const;

	void SwapIp();

	uchar&	operator[](int nPos);

	void Show( ulong ulParams );

public:
	ushort	m_nHardware;
	ushort	m_nProtocol;
	uchar	m_nHeadLen;
	uchar	m_nPacketLen;
	ushort	m_nOpCode;
	uchar	m_aAEtherSrc[6];
	uchar	m_aAIpSrc[4];
	uchar	m_aAEtherDest[6];
	uchar	m_aAIpDest[4];
};

ostream &operator<<(ostream &stream, const CArp& arp);

#endif // __CARP_HEADER__
