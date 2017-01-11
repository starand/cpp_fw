#include "stdafx.h"
#include "ip.h"
#include "macroes.h"
#include "stdutils.h"

using namespace StdUtils;

const char* aszPortocolNames[] = 
{
	"HOPOPT, IPv6 Hop-by-Hop Option",
	"ICMP",
	"IGAP/IGMP/RGMP",
	"GGP, Gateway to Gateway Protocol",
	"IP in IP encapsulation",
	"ST, Internet Stream Protocol",
	"TCP",
	"UCL, CBT",
	"EGP, Exterior Gateway Protocol",
	"IGRP, Interior Gateway Routing Protocol",
	"BBN RCC Monitoring",
	"NVP, Network Voice Protocol",
	"PUP",
	"ARGUS",
	"EMCON, Emission Control Protocol",
	"XNET, Cross Net Debugger",
	"Chaos",
	"UDP",
	"TMux, Transport Multiplexing Protocol",
	"DCN Measurement Subsystems",
	"HMP, Host Monitoring Protocol",
	"Packet Radio Measurement",
	"XEROX NS IDP",
	"Trunk-1",
	"Trunk-2",
	"Leaf-1",
	"Leaf-2",
	"RDP, Reliable Data Protocol",
	"IRTP, Internet Reliable Transaction Protocol",
	"ISO Transport Protocol Class 4",
	"NETBLT, Network Block Transfer",
	"MFE Network Services Protocol",
	"MERIT Internodal Protocol",
	"DCCP, Datagram Congestion Control Protocol",
	"Third Party Connect Protocol",
	"IDPR, Inter-Domain Policy Routing Protocol",
	"XTP, Xpress Transfer Protocol",
	"Datagram Delivery Protocol",
	"IDPR, Control Message Transport Protocol",
	"TP++ Transport Protocol",
	"IL Transport Protocol",
	"IPv6 over IPv4",
	"SDRP, Source Demand Routing Protocol",
	"IPv6 Routing header",
	"IPv6 Fragment header",
	"IDRP, Inter-Domain Routing Protocol",
	"RSVP, Reservation Protocol",
	"GRE, General Routing Encapsulation",
	"DSR, Dynamic Source Routing Protocol",
	"BNA",
	"ESP, Encapsulating Security Payload",
	"AH, Authentication Header",
	"I-NLSP, Integrated Net Layer Security TUBA",
	"SWIPE, IP with Encryption",
	"NARP, NBMA Address Resolution Protocol",
	"Minimal Encapsulation Protocol",
	"TLSP, Transport Layer Security Protocol using Kryptonet key management",
	"SKIP",
	"ICMPv6, Internet Control Message Protocol for IPv6 MLD, Multicast Listener Discovery",
	"IPv6 No Next Header",
	"IPv6 Destination Options",
	"Any host internal protocol",
	"CFTP",
	"Any local network",
	"SATNET and Backroom EXPAK",
	"Kryptolan",
	"MIT Remote Virtual Disk Protocol",
	"Internet Pluribus Packet Core",
	"Any distributed file system",
	"SATNET Monitoring",
	"VISA Protocol",
	"Internet Packet Core Utility",
	"Computer Protocol Network Executive",
	"Computer Protocol Heart Beat",
	"Wang Span Network",
	"Packet Video Protocol",
	"Backroom SATNET Monitoring",
	"SUN ND PROTOCOL-Temporary",
	"WIDEBAND Monitoring",
	"WIDEBAND EXPAK",
	"ISO-IP",
	"VMTP, Versatile Message Transaction Protocol",
	"SECURE-VMTP",
	"VINES",
	"TTP",
	"NSFNET-IGP",
	"Dissimilar Gateway Protocol",
	"TCF",
	"EIGRP",
	"OSPF, Open Shortest Path First Routing Protocol MOSPF, Multicast Open Shortest Path First",
	"Sprite RPC Protocol",
	"Locus Address Resolution Protocol",
	"MTP, Multicast Transport Protocol",
	"AX25",
	"IP-within-IP Encapsulation Protocol",
	"Mobile Internetworking Control Protocol",
	"Semaphore Communications Sec Pro",
	"EtherIP",
	"Encapsulation Header",
	"Any private encryption scheme",
	"GMTP",
	"IFMP, Ipsilon Flow Management Protocol",
	"PNNI over IP",
	"PIM, Protocol Independent Multicast",
	"ARIS",
	"SCPS",
	"QNX",
	"Active Networks",
	"IPPCP, IP Payload Compression Protocol",
	"SNP, Sitara Networks Protocol",
	"Compaq Peer Protocol",
	"IPX in IP",
	"VRRP, Virtual Router Redundancy Protocol",
	"PGM, Pragmatic General Multicast",
	"any 0-hop protocol",
	"L2TP, Level 2 Tunneling Protocol",
	"DDX, D-II Data Exchange",
	"IATP, Interactive Agent Transfer Protocol",
	"ST, Schedule Transfer",
	"SRP, SpectraLink Radio Protocol",
	"UTI",
	"SMP, Simple Message Protocol",
	"SM",
	"PTP, Performance Transparency Protocol",
	"ISIS over IPv4",
	"FIRE",
	"CRTP, Combat Radio Transport Protocol",
	"CRUDP, Combat Radio User Datagram",
	"SSCOPMCE",
	"IPLT",
	"SPS, Secure Packet Shield",
	"PIPE, Private IP Encapsulation within IP",
	"SCTP, Stream Control Transmission Protocol",
	"Fibre Channel",
	"RSVP-E2E-IGNORE",
	"Mobility Header",
	"UDP-Lite, Lightweight User Datagram Protocol",
	"MPLS in IP",
	"MANET Protocols",
	"HIP, Host Identity Protocol",
	"Shim6, Level 3 Multihoming Shim Protocol for IPv6"
};

//////////////////////////////////////////////////////////////
// CIp implementation

ushort CIp::GetHeaderLen() const
{
	return ihl * 4; 
}

ushort CIp::GetLength() const
{
	return ntohs( length );
}

ushort CIp::GetDataLen() const
{
	return GetLength() - GetHeaderLen();
}

uchar& CIp::operator[]( size_t i )
{
	return *((uchar *)this + i);
}

void CIp::CalculateCS()
{
	checksum = 0;
	checksum = NetUtils::CheckSum( (ushort *)this, sizeof(CIp) );
}

const char* CIp::GetSrcAddress() const
{
	return NetUtils::Uint2Ip( isrc );
}	

const char* CIp::GetDestAddress() const
{
	return NetUtils::Uint2Ip( idest );
}

void CIp::SetLength( ushort nLen )
{
	length = htons( nLen );
}

ushort CIp::GetIdent() const 
{
	return ntohs( ident );
}

void CIp::SetIdent( ushort nIdent )
{
	ident = htons( nIdent );
}

void CIp::SetDestAddress( const char* szDestAddress ) 
{ 
	idest = NetUtils::Ip2Uint( szDestAddress );
}

void CIp::SetSrcAddress( const char* szSrcAddress ) 
{ 
	isrc = NetUtils::Ip2Uint( szSrcAddress );
}

void CIp::Show( ulong ulParams ) const
{
	cout << dec << GetProtocol() << " " << GetSrcAddress() << " > " << GetDestAddress() << endl;
}

const char* CIp::GetProtocol() const
{
	static const char szUnknownProtocol[] = "Unknown";
	static const size_t nNamesCount = ARRAY_SIZE( aszPortocolNames );

	if( prot >= nNamesCount ) return szUnknownProtocol;
	else return aszPortocolNames[ prot ];
}

void CIp::SwapIp()
{
	uint nTemp = isrc; isrc = idest; idest = nTemp;
}

ushort CIp::GetCS() const
{
	return ntohs( checksum );
}

uchar* CIp::GetData() const
{
	return (uchar *)((char *)this + GetHeaderLen());
}

ostream &operator<<( ostream &stream, const CIp& ip )
{
	stream << "IP " << ip.GetSrcAddress() << " > " << ip.GetDestAddress() << " " << ip.GetProtocol() << endl;
	return stream;
}

uchar CIp::Protocol() const
{
	return prot;
}
