#include "StdAfx.h"
#include "notify.h"
#include "socket.h"
#include "logerror.h"
#include "consts.h"

//////////////////////////////////////////////////
// CNotify implementation

CNotify::CNotify( const string& sServerAddress /*= DEFAUL_NOTIFIER_SERVER*/, ushort nPort /*= DEFAULT_NOTIFIER_PORT*/ ) 
	: m_nPort(nPort), m_sServerAddress(sServerAddress)
{
}

CNotify::~CNotify()
{
}

bool CNotify::ShowMessage( const string& sMsg, const string& sTitle, uint nType /*= NOTIFY_INFO*/ )
{
	bool bResult = false;

	do
	{
		if( sMsg.empty() ) LOG_ERROR_BREAK( szDataIsEmpty );

		CSocket sockClient(false);
		if( !sockClient.Connect(m_sServerAddress, m_nPort) ) {
			LOG_ERROR3_BREAK(g_szUnableToConnectoTo, m_sServerAddress.c_str(), m_nPort);
		}

		string sMsgPacked = PackMessage( sMsg, sTitle, nType );
		if( !sockClient.SendString(sMsgPacked) ) {
			LOG_ERROR_BREAK(g_szUnableToSendData);
		}

		bResult = true;
	}
	while( false );

	return bResult;
}

bool CNotify::ShowPackedMessage( const string& sMessageData )
{
	START_FUNCTION_BOOL();

	if( sMessageData.empty() ) LOG_ERROR_BREAK( szDataIsEmpty );

	CSocket sockClient;
	if( !sockClient.Connect(m_sServerAddress, m_nPort) ) {
		LOG_ERROR3_BREAK(g_szUnableToConnectoTo, m_sServerAddress.c_str(), m_nPort);
	}

	if( !sockClient.SendString(sMessageData) ) {
		LOG_ERROR_BREAK(g_szUnableToSendData);
	}

	END_FUNCTION_BOOL();
}

string& CNotify::PackMessage( const string& sMsg, const string& sTitle, uint nType )
{
	static string sResult;
	sResult = sMsg + szSeparator + sTitle + szSeparator + IntToStr(nType);
	return sResult;
}

bool CNotify::UnpackMessage( const string& sFullMsg, string& sMsg, string& sTitle, uint& nType )
{
	bool bResult = false;

	string_v vsMessageInfo;
	size_t nCount = StrUtils::split( sFullMsg.c_str(), vsMessageInfo, szSeparator );

	if( 3 == nCount )
	{
		sMsg = vsMessageInfo[ 0 ];
		sTitle = vsMessageInfo[ 1 ];
		nType = atoi( vsMessageInfo[2].c_str() );
		bResult = true;
	}

	return bResult;
}
