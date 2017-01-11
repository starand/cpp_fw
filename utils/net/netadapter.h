#ifndef __INTERFACE__
#define __INTERFACE__

#include "types.h"

#include <pcap.h>
#include "pcap-int.h"
#include <packet32.h>
#include <ntddndis.h>

#pragma comment(lib, "wpcap.lib")
#pragma comment(lib, "packet.lib")

#include "types.h"

class CNetAdapter
{
public:
	CNetAdapter() { }
	CNetAdapter( size_t nDevNum, size_t nTimeOut = 1 );

	void GetDeviceList(string_v& svResult);
	bool IsValidDeviceNumber(size_t nNumber) const;
	size_t GetDeviceCount();

	const char* GetName() const { return m_sName.c_str(); }
	const char* GetShortName() const { return m_sShortName.c_str(); }
	const char* GetDescription() const { return m_sDesc.c_str(); }

	const char* GetDeviceMac() const;
	string GetMacAddr() const;
	const char* GetDeviceIp() const;
	string GetIpAddr() const;
	const char* GetNetMask() const;
	const char* GetBroadAddr() const;

	void* GetPacketData() const;
	size_t GetPacketLength() const;
	const char* GetPacketTime() const;

	bool Open(size_t nDevNum = 1, size_t nTimeOut = 1000);

	void SetIpAddr(uint nIpAddr);
	void SetIpAddr(char* pszIp);
	void SetMacAddr(char* pszMac);

	bool SetFilter(char* pszFilter);
	bool CapturePacket();
	bool CapturePacketTimeout(size_t nTimeout = 0);

	int SendPacket(void* pPacket, size_t nLength) const;
	int SendPacket() const;

	const char* GetLastError() const;

protected:
	bool RetrieveDeviceList();
	bool AssignGetPacketTime();

private:
	pcap_if_t		*m_pDevice, *m_pAllDevs;
	pcap_t*			m_pHandle;

	string			m_sFilter;

	pcap_pkthdr*	m_pHeader;
	void*			m_pPktData;

	size_t			m_nIfaceCount, m_nDevNum;
	size_t			m_nTimeOut;

	tm				m_LTime;
	string			m_sTimeString;

	union {	uint m_nIpAddr; uchar m_aIpAddr[4];	};
	union { uint m_nNetmask; uchar m_aNetmask[4]; };
	union {	uint m_nLocalIp; uchar m_aLocalIp[4]; };	
	union { uint m_nBroadAddr; uchar m_aBroadAddr[4]; };


	uchar			m_aMacAddr[ 6 ], m_aLocalMac[ 6 ];
	string			m_sDesc, m_sName, m_sShortName;

	string			m_sError;
	char			m_ErrorBuf[PCAP_ERRBUF_SIZE]; 
};

#endif // __INTERFACE__
