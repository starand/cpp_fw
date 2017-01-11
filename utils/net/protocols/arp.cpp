#include "stdafx.h"
#include "arp.h"
#include "strutils.h"

#include <WinSock2.h>
#pragma comment( lib, "ws2_32.lib" )

uchar& CArp::operator[](int nPos)
{
	return *((uchar *)this + nPos);
}

ushort CArp::OpCode() const
{ 
	return ntohs(m_nOpCode); 
}

const char* CArp::GetEtherDest() const
{
	return StrUtils::MacAddr(m_aAEtherDest);
}

const char* CArp::GetEtherSrc() const
{
	return StrUtils::MacAddr(m_aAEtherSrc);
}

const char* CArp::GetIpSrc() const
{
	return StrUtils::IpAddr(m_aAIpSrc);
}

const char* CArp::GetIpDest() const
{
	return StrUtils::IpAddr(m_aAIpDest);
}

void CArp::SetEtherDest(char* pszDest)
{
	int a, b, c, x, y, z;
	sscanf_s(pszDest, "%2X:%2X:%2X:%2X:%2X:%2X", &a, &b, &c, &x, &y, &z);
	m_aAEtherDest[0] = a; m_aAEtherDest[1] = b; m_aAEtherDest[2] = c;
	m_aAEtherDest[3] = x; m_aAEtherDest[4] = y; m_aAEtherDest[5] = z;
}

void CArp::SetEtherSrc(char* pszSrc)
{
	int a, b, c, x, y, z;
	sscanf_s(pszSrc, "%2X:%2X:%2X:%2X:%2X:%2X", &a, &b, &c, &x, &y, &z);
	m_aAEtherSrc[0] = a; m_aAEtherSrc[1] = b; m_aAEtherSrc[2] = c;
	m_aAEtherSrc[3] = x; m_aAEtherSrc[4] = y; m_aAEtherSrc[5] = z;
}

void CArp::SetIpDest(char* pszDest) 
{ 
	int a, b, c, d;
	sscanf_s(pszDest, "%i.%i.%i.%i", &a, &b, &c, &d);
	m_aAIpDest[0] = a; m_aAIpDest[1] = b; m_aAIpDest[2] = c; m_aAIpDest[3] = d;
}

void CArp::SetIpSrc(char* pszSrc) 
{ 
	int a, b, c, d;
	sscanf_s(pszSrc, "%i.%i.%i.%i", &a, &b, &c, &d); 
	m_aAIpSrc[0] = a; m_aAIpSrc[1] = b; m_aAIpSrc[2] = c; m_aAIpSrc[3] = d;
}

string CArp::GetHardware()
{
	string sResult;

	switch(ntohs(m_nHardware))
	{
	case 0:  sResult.assign("reserved."); break;
	case 1:  sResult.assign("Ethernet."); break;
	case 2:  sResult.assign("Experimental Ethernet."); break;
	case 3:  sResult.assign("Amateur Radio AX.25."); break;
	case 4:  sResult.assign("Proteon ProNET Token Ring."); break;
	case 5:  sResult.assign("Chaos."); break;
	case 6:  sResult.assign("IEEE 802."); break;
	case 7:  sResult.assign("ARCNET."); break;
	case 8:  sResult.assign("Hyperchannel."); break;
	case 9:  sResult.assign("Lanstar."); break;
	case 10: sResult.assign("Autonet Short Address."); break;
	case 11: sResult.assign("LocalTalk."); break;
	case 12: sResult.assign("LocalNet (IBM PCNet or SYTEK LocalNET)."); break;
	case 13: sResult.assign("Ultra link."); break;
	case 14: sResult.assign("SMDS."); break;
	case 15: sResult.assign("Frame Relay."); break;
	case 16: sResult.assign("ATM, Asynchronous Transmission Mode."); break;
	case 17: sResult.assign("HDLC."); break;
	case 18: sResult.assign("Fibre Channel."); break;
	case 19: sResult.assign("ATM, Asynchronous Transmission Mode."); break;
	case 20: sResult.assign("Serial Line."); break;
	case 21: sResult.assign("ATM, Asynchronous Transmission Mode."); break;
	case 22: sResult.assign("MIL-STD-188-220."); break;
	case 23: sResult.assign("Metricom."); break;
	case 24: sResult.assign("IEEE 1394.1995."); break;
	case 25: sResult.assign("MAPOS."); break;
	case 26: sResult.assign("Twinaxial."); break;
	case 27: sResult.assign("EUI-64."); break;
	case 28: sResult.assign("HIPCArp."); break;
	case 29: sResult.assign("IP and CArp over ISO 7816-3."); break;
	case 30: sResult.assign("CArpSec."); break;
	case 31: sResult.assign("IPsec tunnel."); break;
	case 32: sResult.assign("Infiniband."); break;
	case 33: sResult.assign("CAI, TIA-102 Project 25 Common Air Interface."); break;
	case 34: sResult.assign("Wiegand Interface."); break;
	case 35: sResult.assign("Pure IP."); break;
	case 36: sResult.assign("HW_EXP1"); break;
	case 256:sResult.assign("HW_EXP2"); break;
	default:sResult.assign("reserved");
	};

	return sResult;
}

string CArp::GetOpcode() const
{
	string sResult;

	switch(ntohs(m_nOpCode))
	{
	case 1: sResult.assign("Request"); break;
	case 2: sResult.assign("Reply"); break;
	case 3: sResult.assign("Request Reverse"); break;
	case 4: sResult.assign("Reply Reverse"); break;
	case 5: sResult.assign("DRCARP Request"); break;
	case 6: sResult.assign("DRCARP Reply"); break;
	case 7: sResult.assign("DRCARP Error"); break;
	case 8: sResult.assign("InCARP Request"); break;
	case 9: sResult.assign("InCARP Reply"); break;
	case 10: sResult.assign("CARP NAK"); break;
	case 11: sResult.assign("MARS Request"); break;
	case 12: sResult.assign("MARS Multi"); break;
	case 13: sResult.assign("MARS MServ"); break;
	case 14: sResult.assign("MARS Join"); break;
	case 15: sResult.assign("MARS Leave"); break;
	case 16: sResult.assign("MARS NAK"); break;
	case 17: sResult.assign("MARS Unserv"); break;
	case 18: sResult.assign("MARS SJoin"); break;
	case 19: sResult.assign("MARS SLeave"); break;
	case 20: sResult.assign("MARS Grouplist Request"); break;
	case 21: sResult.assign("MARS Grouplist Reply"); break;
	case 22: sResult.assign("MARS Redirect Map"); break;
	case 23: sResult.assign("MAPOS UNCARP"); break;
	case 24: sResult.assign("OP_EXP1"); break;
	case 25: sResult.assign("OP_EXP2"); break;
	default: sResult.assign("reserved");
	};

	return sResult;
}

void CArp::SwapIp()
{
	uchar tmp[4];
	memcpy(tmp, m_aAIpSrc, 4);
	memcpy(m_aAIpSrc, m_aAIpDest, 4);
	memcpy(m_aAIpDest, tmp, 4);
}

ostream &operator<<(ostream &stream, const CArp& arp)
{
	stream << "ARP " << arp.GetIpSrc() << " > " << arp.GetIpDest() << " " << arp.GetOpcode();
	return stream;
}

void CArp::Show( ulong ulParams )
{
	cout << "ARP " << GetIpSrc() << " > " << GetIpDest() << " " << GetOpcode() << endl;
}

