#ifndef __CIP_HEADER__
#define __CIP_HEADER__

#include "netutils.h"

#define IP_TCP		6
#define IP_UDP		17

class CIp
{
public:
	ushort GetHeaderLen() const;
	ushort GetLength() const;
	ushort GetDataLen() const;

	uchar &operator[]( size_t i );
	
	void CalculateCS();

	const char* GetSrcAddress() const;
	const char* GetDestAddress() const;

	
	void SetLength( ushort nLen );
	ushort GetIdent() const;
	void SetIdent( ushort nIdent );

	void SetDestAddress( const char* szDestAddress );
	void SetSrcAddress( const char* szSrcAddress ) ;

	void Show( ulong ulParams ) const;
	const char* GetProtocol() const;
	uchar Protocol() const;
	
	void SwapIp();

	ushort GetCS() const;

	uchar* GetData() const;

private:
			uchar	ihl:4;
	uchar	ver:4;
	uchar	tos;
	ushort	length;
	ushort	ident;
	ushort	fl_offset;
	uchar	TTL;
	uchar	prot;
	ushort	checksum;
	union
	{
		uchar	src[4];	
		uint	isrc;
	};
	union 
	{
		uchar	dest[4];
		uint	idest;
	};	
};

ostream &operator<<( ostream &stream, const CIp& ip );

#endif // __CIP_HEADER__
