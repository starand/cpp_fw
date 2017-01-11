#ifndef __H_SOUND__
#define __H_SOUND__

class CSound
{
public:
	CSound( const string& sDeviceAlias = "MY_DEVICE" );

	bool Play( const string& sSoundFile, bool bWait = false );
	bool Stop();
	bool Close();
	bool Pause();
	bool Resume();

	bool LeftOn();
	bool LeftOff();
	bool RightOn();
	bool RightOff();

	bool VolPlus();
	bool VolMinus();
	bool Mute();
	bool UnMute();

	const char* GetStatus();

	size_t GetLength();

	bool IsOpen() const { return m_bOpen; }
	const char* GetTrackFile() const { return m_sSoundFile.c_str(); }

	const char* GetDeviceAlias() const { return m_sDeviceAlias.c_str(); }
	const char* GetLastDeviceCommand() const { return m_sDeviceCommand.c_str(); }

	bool IsPlaying() const { return m_bIsPlaying; }
	bool IsPause() const { return m_bPause; }

	static bool PlaySound( const string& sSoundFile, bool bWait = false );

protected:
	bool SetVolume( size_t nVolume );
	bool SetLeftVolume( size_t nVolume );
	bool SetRightVolume( size_t nVolume );

	virtual void on_volume_chnaged(size_t nVolume) { }

private:	
	string	m_sDeviceAlias;
	string	m_sDeviceCommand;

	string	m_sSoundFile;
	bool	m_bOpen;
	bool	m_bIsPlaying;
	bool	m_bPause;

	size_t	m_nVolumeLevel;
};

#endif // __H_PLAYSOUND__
