#ifndef __CCTcp_HEADER__
#define __CCTcp_HEADER__

#include "types.h"


#define	T_FIN	0x01
#define	T_SYN	0x02
#define	T_RST	0x04
#define	T_PUSH	0x08
#define	T_ACK	0x10
#define	T_URG	0x20

class CIp;
// psevdo header for tcp checksum
struct PH
{ 
	uint src;
	uint dest; 
	uchar mbz;
	uchar prot; 
	ushort pkt_len;
};

class CTcp
{
public:
	ushort GetDataLen( CIp* pIpHeader ) const;
	ushort GetHeaderLen() const;

	ushort GetOffset() const;
	void SetOffset( ushort nOffset );

	uchar &operator[]( size_t i );

	ushort GetSrcPort() const;
	void SetSrcPort( ushort nPort );
	ushort GetDestPort() const;
	void SetDesrPort( ushort nPort );

	uint GetSN() const;
	void SetSN( uint uiSN );
	uint GetAN() const;
	void SetAN( uint uiAN );

	ushort GetWindow() const;
	void SetWindow( ushort usWindow );

	void Show( CIp* pIpHeader, ulong ulParams ) const;
	
	ushort GetUrgPoint() const;
	ushort GetCS() const;

	uchar* GetData() const;

	const char* GetFlagsString() const;

	void SwapPort();

public:
	ushort sport;
	ushort dport;
	uint sn;
	uint an;
	uchar offset;
	uchar fl;
	ushort window;
	ushort cs;
	ushort urg_point;
};

#endif // __CCTcp_HEADER__
