#include "StdAfx.h"
#include "sound.h"
#include "strutils.h"
#include <mciapi.h>

#pragma comment( lib, "winmm.lib" )

#define MAX_VOLUME		1000
#define MIDDLE_VOLUME	500
#define VOLUME_STEP		50

//////////////////////////////////////////////////
// CSound implementation

CSound::CSound( const string& sDeviceAlias ) 
	: m_sDeviceAlias(sDeviceAlias), m_bOpen(), m_bIsPlaying(), m_bPause(), m_nVolumeLevel(MIDDLE_VOLUME)
{
}

bool CSound::Play( const string& sSoundFile, bool bWait /*= false*/ )
{
	START_FUNCTION_BOOL();

	if( !m_sDeviceAlias.empty() && !m_sSoundFile.empty() ) {
		StrUtils::FormatString( m_sDeviceCommand, "close %s", m_sDeviceAlias.c_str() );
		mciSendStringA( m_sDeviceCommand.c_str(), NULL, 0, 0 );
	}

	m_sSoundFile = sSoundFile;

	StrUtils::FormatString( m_sDeviceCommand, "open \"%s\" alias %s", m_sSoundFile.c_str(), m_sDeviceAlias.c_str() );
	m_bOpen = ( 0 == mciSendStringA( m_sDeviceCommand.c_str(), NULL, 0, 0) );
	if( !m_bOpen ) break;

	SetVolume( m_nVolumeLevel );

	m_bIsPlaying = true;
	StrUtils::FormatString( m_sDeviceCommand, "play %s %s", m_sDeviceAlias.c_str(), (bWait ? "wait" : "") );
	if( 0 != mciSendStringA( m_sDeviceCommand.c_str(), NULL, 0, 0) ) break;

	END_FUNCTION_BOOL();
}

bool CSound::Stop()
{
	StrUtils::FormatString( m_sDeviceCommand, "stop %s", m_sDeviceAlias.c_str() );
	m_bOpen = m_bIsPlaying = false;
	return ( 0 == mciSendStringA( m_sDeviceCommand.c_str(), NULL, 0, 0) );
}

bool CSound::Close()
{
	StrUtils::FormatString( m_sDeviceCommand, "close %s", m_sDeviceAlias.c_str() );
	m_bOpen = m_bIsPlaying = false;
	return ( 0 == mciSendStringA( m_sDeviceCommand.c_str(), NULL, 0, 0) );
}

bool CSound::Pause()
{
	StrUtils::FormatString( m_sDeviceCommand, "pause %s", m_sDeviceAlias.c_str() );
	return m_bPause = ( 0 == mciSendStringA( m_sDeviceCommand.c_str(), NULL, 0, 0) );
}

bool CSound::Resume()
{
	m_bPause = false;
	StrUtils::FormatString( m_sDeviceCommand, "resume %s", m_sDeviceAlias.c_str() );
	return ( 0 == mciSendStringA( m_sDeviceCommand.c_str(), NULL, 0, 0) );
}

bool CSound::SetVolume( size_t nVolume  ) // region between 0 and 1000
{
	ASSERT(IN_RANGE(nVolume, 0, 1000));
	StrUtils::FormatString( m_sDeviceCommand, "setaudio %s volume to %u", m_sDeviceAlias.c_str(), nVolume );
	
	bool result = (0 == mciSendStringA( m_sDeviceCommand.c_str(), NULL, 0, 0));

	if (result)
	{
		on_volume_chnaged(nVolume);
	}
	
	return result;
}

bool CSound::SetLeftVolume( size_t nVolume )
{
	StrUtils::FormatString( m_sDeviceCommand, "setaudio %s left volume to %u", m_sDeviceAlias.c_str(), nVolume );
	return ( 0 == mciSendStringA( m_sDeviceCommand.c_str(), NULL, 0, 0) );
}

bool CSound::SetRightVolume( size_t nVolume )
{
	StrUtils::FormatString( m_sDeviceCommand, "setaudio %s right volume to %u", m_sDeviceAlias.c_str(), nVolume );
	return ( 0 == mciSendStringA( m_sDeviceCommand.c_str(), NULL, 0, 0) );
}

bool CSound::LeftOn()
{
	StrUtils::FormatString( m_sDeviceCommand, "setaudio %s left on", m_sDeviceAlias.c_str() );
	return ( 0 == mciSendStringA( m_sDeviceCommand.c_str(), NULL, 0, 0) );
}

bool CSound::LeftOff()
{
	StrUtils::FormatString( m_sDeviceCommand, "setaudio %s left off", m_sDeviceAlias.c_str() );
	return ( 0 == mciSendStringA( m_sDeviceCommand.c_str(), NULL, 0, 0) );
}

bool CSound::RightOn()
{
	StrUtils::FormatString( m_sDeviceCommand, "setaudio %s right on", m_sDeviceAlias.c_str() );
	return ( 0 == mciSendStringA( m_sDeviceCommand.c_str(), NULL, 0, 0) );
}

bool CSound::RightOff()
{
	StrUtils::FormatString( m_sDeviceCommand, "setaudio %s right off", m_sDeviceAlias.c_str() );
	return ( 0 == mciSendStringA( m_sDeviceCommand.c_str(), NULL, 0, 0) );
}

bool CSound::VolPlus()
{
	bool bResult = false;
	if( m_nVolumeLevel <= MAX_VOLUME - VOLUME_STEP )
	{
		m_nVolumeLevel += VOLUME_STEP;
		bResult = SetVolume( m_nVolumeLevel );
	}
	return bResult;
}

bool CSound::VolMinus()
{
	bool bResult = false;
	if( m_nVolumeLevel >= VOLUME_STEP )
	{
		m_nVolumeLevel -= VOLUME_STEP;
		bResult = SetVolume( m_nVolumeLevel );
	}
	return bResult;
}

bool CSound::Mute()
{
	return SetVolume(0);
}

bool CSound::UnMute()
{
	return SetVolume(m_nVolumeLevel);
}


bool CSound::PlaySound( const string& sSoundFile, bool bWait /*= false*/ )
{
	bool bResult = false;
	if( !sSoundFile.empty() )
	{
		string sPlayCommand = string( "play " ) + sSoundFile + ( bWait ? " wait" : "" );
		bResult = ( 0 == mciSendStringA( sPlayCommand.c_str(), NULL, 0, 0 ) );
	}
	return bResult;
}

size_t CSound::GetLength()
{
	size_t nResult = -1;

	static const char nBufferLen = 16;
	static char szBuffer[ nBufferLen ] = {};

	StrUtils::FormatString( m_sDeviceCommand, "status %s length", m_sDeviceAlias.c_str() );
	if( 0 == mciSendStringA( m_sDeviceCommand.c_str(), szBuffer, nBufferLen, 0) ) {
		nResult = atol( szBuffer );
	}
	
	return nResult;
}

const char* CSound::GetStatus()
{
	static const size_t nBufferSize = 100;
	static char szBuffer[ nBufferSize ] = {};

	StrUtils::FormatString( m_sDeviceCommand, "status %s mode wait", m_sDeviceAlias.c_str() );
	/*int nResult = */mciSendStringA( m_sDeviceCommand.c_str(), szBuffer, nBufferSize, 0);

	return szBuffer;
}
