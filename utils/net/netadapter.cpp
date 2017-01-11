#include "stdafx.h"
#include "netadapter.h"
#include "strutils.h"

const char szErrorAdapterNumberOutOfRange[] = "Interface number out of range";
const char szErrorAdpaterNotFound[] = "Adapter not found";

// search devices and store it to m_pAllDevs
bool CNetAdapter::RetrieveDeviceList() 
{
	if( pcap_findalldevs(&m_pAllDevs, m_ErrorBuf) == -1 )  return FALSE;
	else return TRUE;
}

// retrieve device list 
void CNetAdapter::GetDeviceList( string_v& svResult )
{	
	size_t nCount = 0;
	string sResItem;

	do 
	{
		svResult.clear();

		if( !RetrieveDeviceList() )  break;

		for( m_pDevice = m_pAllDevs; m_pDevice; m_pDevice = m_pDevice->next )
		{
			++nCount;
			sResItem.assign( m_pDevice->name );

			if( m_pDevice->description )
			{
				sResItem.append(" ").append( m_pDevice->description );
			}
			svResult.push_back( sResItem );
		}
	} 
	while( false );
}

// return count of devices
size_t CNetAdapter::GetDeviceCount()
{		
	size_t nDevCount = 0;

	do 
	{
		if( !RetrieveDeviceList() ) break;

		for( m_pDevice = m_pAllDevs; m_pDevice; m_pDevice = m_pDevice->next, ++nDevCount );
	} 
	while( false );

	return ( m_nIfaceCount = nDevCount );
}

// initialize adapter
bool CNetAdapter::Open( size_t nDevNum /* = 1 */, size_t nTimeOut /* = 1000 */ )
{
	bool bResult = false;

	do 
	{
		if( nDevNum > GetDeviceCount() ) 
		{
			m_sError.assign( szErrorAdapterNumberOutOfRange );
			break;
		}

		if( m_nIfaceCount == 0 ) 
		{
			m_sError.assign( szErrorAdpaterNotFound );
			break;
		}

		if(nDevNum < 1 || nDevNum > m_nIfaceCount)
		{
			m_sError.assign( szErrorAdapterNumberOutOfRange);
			break;
		}

		m_nTimeOut = nTimeOut;
		m_nDevNum = nDevNum;

		size_t i = 0;
		for(m_pDevice = m_pAllDevs; i < nDevNum-1; m_pDevice = m_pDevice->next, i++);

		if((m_pHandle = pcap_open_live(m_pDevice->name, 65536, 1, nTimeOut, m_ErrorBuf)) == NULL)
		{
			m_sError.assign( StrUtils::FormatString("Unable to open the adapter. is not supported by WinPcap%s ", m_pDevice->name) );
			break;
		}

		if( m_pDevice->description )
		{
			m_sDesc.assign( m_pDevice->description );
		}

		m_sName.assign( m_pDevice->name );
		m_sShortName = m_sName.substr( 0, m_sName.find('{') );

		// retrieve ip, netmask, 
		if( m_pDevice->addresses != NULL )
		{
			if( m_pDevice->addresses->netmask ) 
			{
				m_nNetmask = ((sockaddr_in *)m_pDevice->addresses->netmask)->sin_addr.S_un.S_addr;
			}
			else m_nNetmask = 0xffffff;

			if( m_pDevice->addresses->addr ) 
			{
				m_nIpAddr = m_nLocalIp = ((struct sockaddr_in *)m_pDevice->addresses->addr)->sin_addr.s_addr;
			}
			if( m_pDevice->addresses->broadaddr )
			{
				m_nBroadAddr = ((struct sockaddr_in *)m_pDevice->addresses->broadaddr)->sin_addr.s_addr;
			}
		}
		else 
		{
			m_nNetmask = m_nIpAddr = m_nLocalIp = m_nBroadAddr = 0;
			m_sError.assign( "Can\'t obtain ip address" );
		}

		// retrieve mac address
		PPACKET_OID_DATA    pOidData;
		CHAR pAddr[ sizeof(PACKET_OID_DATA) + 5 ];
		ZeroMemory( pAddr, sizeof(PACKET_OID_DATA) + 5 );
		pOidData = (PPACKET_OID_DATA)pAddr;
		pOidData->Oid = OID_802_3_CURRENT_ADDRESS;
		pOidData->Length = 6;

		if( PacketRequest(m_pHandle->adapter, FALSE, pOidData) )
		{
			memcpy(m_aLocalMac, pOidData->Data, 6);
			memcpy(m_aMacAddr, pOidData->Data, 6);
		}
		else
		{
			m_sError.assign( "Can\'t obtain address" );
		}

		// WTF ?? --- if( m_pDevice != m_pAllDevs ) break;

		bResult = true;
	} 
	while(false);

	if( !bResult )
	{
		pcap_freealldevs(m_pAllDevs);
	}

	return bResult;
}

CNetAdapter::CNetAdapter( size_t nDevNum, size_t nTimeOut /* = 1 */ )
{
	Open( nDevNum, nTimeOut );
}

// return native network adapter physical address
const char* CNetAdapter::GetDeviceMac() const
{
	static string sDeviceMAC = StrUtils::MacAddr( m_aLocalMac );
	return sDeviceMAC.c_str();
}

// return physical address which was assigned by user
string CNetAdapter::GetMacAddr() const
{
	return StrUtils::MacAddr( m_aMacAddr );
}

// return IP address which was assigned by OS
const char* CNetAdapter::GetDeviceIp() const
{
	static string sDeviceIP= StrUtils::IpAddr( m_aIpAddr );
	return sDeviceIP.c_str();
}

// return IP address which was assigned by user
string CNetAdapter::GetIpAddr() const
{
	return StrUtils::IpAddr( m_aIpAddr );
}

const char* CNetAdapter::GetNetMask() const
{
	static string sNetMask = StrUtils::IpAddr( m_aNetmask );
	return sNetMask.c_str();
}

const char* CNetAdapter::GetBroadAddr() const
{
	static string sBroadCastAddr = StrUtils::IpAddr( m_aBroadAddr );
	return sBroadCastAddr.c_str();
}

void CNetAdapter::SetIpAddr( uint nIpAddr )
{
	m_nIpAddr = nIpAddr;
}

void CNetAdapter::SetIpAddr(char* pszIp)
{
	sscanf_s( pszIp, "%d.%d.%d.%d",	&m_aIpAddr[0],&m_aIpAddr[1],&m_aIpAddr[2],&m_aIpAddr[3] );
}

void CNetAdapter::SetMacAddr(char* pszMac)
{
	sscanf_s( pszMac, "%2X:%2X:%2X:%2X:%2X:%2X", 
		&m_aMacAddr[0],&m_aMacAddr[1],&m_aMacAddr[2],&m_aMacAddr[3],&m_aMacAddr[4],&m_aMacAddr[5] );
	
}

void* CNetAdapter::GetPacketData() const
{
	return m_pPktData;
}

size_t CNetAdapter::GetPacketLength() const
{
	if( m_pHeader ) return m_pHeader->len;
	return 0;
}

const char* CNetAdapter::GetPacketTime() const
{
	return m_sTimeString.c_str();
}

bool CNetAdapter::SetFilter( char* pszFilter )
{
	bool bResult = false;

	do 
	{
		bpf_program bpfCode;

		//compile the filter
		if( pcap_compile(m_pHandle, &bpfCode, pszFilter, 1, m_nNetmask) < 0 )
		{
			m_sError.assign( "Unable to compile the packet filter. Check the syntax." );
			break;
		}

		//set the filter
		if( pcap_setfilter(m_pHandle, &bpfCode) < 0 )
		{
			m_sError.assign( "Error setting the filter" );
			break;
		}

		bResult = true;
	} 
	while (false);

	if( !bResult )
	{
		pcap_freealldevs(m_pAllDevs);
	}

	return bResult;
}

bool CNetAdapter::AssignGetPacketTime()
{
	bool bResult = false;

	do 
	{
		char szTimeStr[20] = { 0 };

		if( !m_pHeader ) break;

		time_t local_tv_sec = m_pHeader->ts.tv_sec;
		localtime_s( &m_LTime, &local_tv_sec );

		strftime( szTimeStr, sizeof(szTimeStr), "%H:%M:%S", &m_LTime );
		m_sTimeString.assign( szTimeStr );
		
		bResult = true;
	} 
	while (false);

	return bResult;	
}

bool CNetAdapter::CapturePacket()
{
	bool bResult = false;

	do 
	{
		int nResult = 0;
		while( (nResult = pcap_next_ex(m_pHandle, &m_pHeader, (const uchar **)&m_pPktData)) == 0 );

		if( nResult == -1 )
		{
			m_sError.assign( StrUtils::FormatString("Error reading the packets: %s", pcap_geterr(m_pHandle)) );
			break;
		}
		
		AssignGetPacketTime();
		bResult = true;
	} 
	while (false);

	return bResult;
}

bool CNetAdapter::CapturePacketTimeout(size_t nTimeout /* = 0 */)
{
	bool bResult = false;

	Sleep( nTimeout );
	if( pcap_next_ex(m_pHandle, &m_pHeader, (const uchar **)&m_pPktData) > 0 )
	{
		AssignGetPacketTime();
		bResult = true;
	}

	return bResult;
}

bool CNetAdapter::IsValidDeviceNumber(size_t nNumber) const
{
	if( nNumber < 1 || nNumber > m_nIfaceCount ) return false;
	else return true;
}

int CNetAdapter::SendPacket(void* pPacket, size_t nLength) const
{
	return pcap_sendpacket( m_pHandle, (const u_char*)pPacket, nLength );
}

int CNetAdapter::SendPacket() const
{
	return pcap_sendpacket( m_pHandle, (u_char*)m_pPktData, m_pHeader->len );
}

const char* CNetAdapter::GetLastError() const
{
	return m_sError.c_str();
}
