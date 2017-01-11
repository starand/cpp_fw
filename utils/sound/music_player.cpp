#include "stdafx.h"
#include "music_player.h"
#include "sound.h"

#define CHECK_TIMEOUT 200

static string g_sEmptyString;

//////////////////////////////////////////////////
// CMusicPlayer implementation

music_player_t::music_player_t( const string& sDeviceAlias /*= DEFAULT_DEVICE_ALIAS*/ ) 
	: CSound(sDeviceAlias)
	, m_nCurrentTrack()
	, m_empcState(EMPC_NONE)
	, m_wait_event(false, false)
{
	start(); // start checker thread - it checks when tracks is end
}

music_player_t::~music_player_t()
{
	stop();
}

void music_player_t::SetTrackList( const string_v& vsList )
{
	m_vsTrackList = vsList;
	m_nCurrentTrack = 0;
}

void music_player_t::SetCurrentTrackNumber( size_t nTrackID )
{
	m_nCurrentTrack = ( m_vsTrackList.size() > nTrackID ? nTrackID : 0 );
}

int music_player_t::GetCurrentTrackNumber() const
{
	return m_nCurrentTrack;
}

string& music_player_t::GetCurrentTrack()
{
	return m_vsTrackList.size() ? m_vsTrackList[m_nCurrentTrack] : g_sEmptyString;
}

string music_player_t::GetTrackName()
{
	string& currenTrack = GetCurrentTrack();
	string result;

	if (!currenTrack.empty())
	{
		sizeint lastSlahPos = currenTrack.find_last_of("\\/");
		if (string::npos == lastSlahPos)
		{
			result = currenTrack;
		}
		else
		{
			result = currenTrack.substr(++lastSlahPos);
		}
	}

	return result;
}


void music_player_t::ExecuteCommand( EMPCOMMAND empcComamnd ) 
{ 
	m_empcState = empcComamnd; 
}


void music_player_t::add_tracker(musicplayer_tracker_inf_t *tracker)
{
	ASSERT(tracker != NULL);
	m_trackers.insert(tracker);
}

void music_player_t::remove_tracker(musicplayer_tracker_inf_t *tracker)
{
	ASSERT(tracker != NULL);
	m_trackers.erase(tracker);
}


void music_player_t::on_track_changed(size_t track)
{
	musicplayer_tracker_list_t::const_iterator iter = m_trackers.begin();

	for (; iter != m_trackers.end(); ++iter)
	{
		(*iter)->on_track_changed(track);
	}
}


/*virtual */
void music_player_t::do_run()
{
	while( true )
	{
		m_wait_event.wait(CHECK_TIMEOUT);
		if (is_stopping())
		{
			break;
		}

		CheckCommands();
		CheckTrackIsEnd();
	}
}

/*virtual */
void music_player_t::do_stop()
{
	m_wait_event.set();
}


void music_player_t::CheckCommands()
{
	switch( m_empcState )
	{
	case EMPC_PLAYPAUSE:
		if( !IsPlaying() )
		{
			if( m_vsTrackList.size() )
				Play( m_vsTrackList[m_nCurrentTrack] );
		}
		else
		{
			if( IsPause() ) CSound::Resume();
			else Pause();
		}
		break;
	case EMPC_CUSTOMTRACK:
		Play(m_vsTrackList[m_nCurrentTrack]);
		on_track_changed(m_nCurrentTrack);
		break;
	case EMPC_FORWARD:
		if( IsPlaying() && !IsPause() )
		{
			if( ++m_nCurrentTrack == m_vsTrackList.size() ) m_nCurrentTrack = 0;
			Play(m_vsTrackList[m_nCurrentTrack]);
			on_track_changed(m_nCurrentTrack);
		}
		break;
	case EMPC_BACKWARD:
		if( IsPlaying() && !IsPause() )
		{
			if( m_nCurrentTrack == 0 ) m_nCurrentTrack = m_vsTrackList.size() - 1;
			Play(m_vsTrackList[--m_nCurrentTrack]);
			on_track_changed(m_nCurrentTrack);
		}
		break;
	case EMPC_VOLPLUS:
		VolPlus();
		break;
	case EMPC_VOLMINUS:
		VolMinus();
		break;
	case EMPC_MUTE:
		Mute();
		break;
	case EMPC_UNMUTE:
		UnMute();
		break;
	}

	m_empcState = EMPC_NONE;
}

void music_player_t::CheckTrackIsEnd()
{
	static const char szStopped[] = "stopped";
	if( IsPlaying() && !IsPause() && stricmp(GetStatus(), szStopped) == 0 ) {
		m_empcState = EMPC_FORWARD;
	}
}

void music_player_t::PlayMusic()
{
	if (!GetIsPlaying())
	{
		ExecuteCommand(EMPC_PLAYPAUSE);
	}
}

void music_player_t::PauseMusic()
{
	if (GetIsPlaying())
	{
		ExecuteCommand(EMPC_PLAYPAUSE);
	}
}

void music_player_t::PlayPause()
{
	ExecuteCommand(EMPC_PLAYPAUSE);
}


void music_player_t::set_volume(size_t level)
{
	SetVolume(level);
}
