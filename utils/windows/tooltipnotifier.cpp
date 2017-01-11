#include "StdAfx.h"
#include "tooltipnotifier.h"

//////////////////////////////////////////////////
// CToolTipNotifier implementation

CToolTipNotifier::CToolTipNotifier()
{
	memset( &m_nidIconData, 0, sizeof(m_nidIconData) ); 
	m_nidIconData.cbSize = sizeof( m_nidIconData );

#ifdef _CONSOLE
	m_nidIconData.hWnd = GetConsoleWindow();
#else
	m_nidIconData.hWnd = NULL; _ASSERT( false ); // not implemented yet
#endif
}

CToolTipNotifier::~CToolTipNotifier()
{
	Destroy();
	if( m_nidIconData.hIcon )
	{
		DestroyIcon( m_nidIconData.hIcon );
	}
}

bool CToolTipNotifier::SetIcon( const char* szIconFile, const char* szIconTip /*= NULL*/ )
{
	bool bResult = false;

	do
	{
		if( !szIconFile ) break;

		m_nidIconData.uFlags |= NIF_ICON;
		m_nidIconData.hIcon = (HICON)LoadImageA( NULL, szIconFile, IMAGE_ICON, 32, 32, LR_LOADFROMFILE );

		if( szIconTip )
		{
			strncpy_s( m_nidIconData.szTip, 127, szIconTip, 127 );
			m_nidIconData.uFlags |= NIF_TIP;
		}

		bResult = true;
	}
	while( false );

	return bResult;
}

bool CToolTipNotifier::SetMessage( const char* szMessage, const char* szTitle, uint uiFLags /*= NIIF_INFO*/ )
{
	bool bResult = false;

	do
	{
		if( !szMessage || !szTitle ) break;

		m_nidIconData.uFlags |= NIF_INFO;
		m_nidIconData.dwInfoFlags = uiFLags;

		strncpy_s( m_nidIconData.szInfo, 255, szMessage, 255 );
		strncpy_s( m_nidIconData.szInfoTitle, 63, szTitle, 63 );

		bResult = true;
	}
	while( false );

	return bResult;
}

bool CToolTipNotifier::Create()
{
	return ( TRUE == Shell_NotifyIcon(NIM_ADD, &m_nidIconData) );
}

bool CToolTipNotifier::Update()
{
	return ( TRUE == Shell_NotifyIcon(NIM_MODIFY, &m_nidIconData) );
}

bool CToolTipNotifier::Destroy()
{
	return ( TRUE == Shell_NotifyIcon(NIM_DELETE, &m_nidIconData) );
}
