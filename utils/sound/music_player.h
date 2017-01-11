#ifndef __H_MUSICPLAYER__
#define __H_MUSICPLAYER__

#include "sound.h"
#include <threading.h>
#include <set>


#define DEFAULT_DEVICE_ALIAS "SMART_PLAYER"

enum EMPCOMMAND
{
	EMPC__MIN = 0,
	EMPC_NONE = EMPC__MIN,

	EMPC_PLAYPAUSE,
	EMPC_CUSTOMTRACK,
	EMPC_FORWARD,
	EMPC_BACKWARD,
	EMPC_VOLPLUS,
	EMPC_VOLMINUS,
	EMPC_MUTE,
	EMPC_UNMUTE,

	EMPC__MAX,
};


class musicplayer_tracker_inf_t
{
public:
	virtual ~musicplayer_tracker_inf_t() { }

	virtual void on_track_changed(size_t track) = 0;
};

//////////////////////////////////////////////////
// CMusicPlayer declaration

class music_player_t : private CSound, private thread_base_t
{
public:
	music_player_t( const string& sDeviceAlias = DEFAULT_DEVICE_ALIAS );
	~music_player_t();

	void SetTrackList( const string_v& vsList );

	void SetCurrentTrackNumber( size_t nTrackID );
	int GetCurrentTrackNumber() const;

	string& GetCurrentTrack();
	string GetTrackName();

	void ExecuteCommand(EMPCOMMAND empcComamnd);

public:
	void add_tracker(musicplayer_tracker_inf_t *tracker);
	void remove_tracker(musicplayer_tracker_inf_t *tracker);

private:
	void on_track_changed(size_t track);

public:
	bool GetIsPlaying() const { return IsPlaying()&&  !IsPause(); }
	void PlayMusic();
	void PauseMusic();
	void PlayPause();

	void set_volume(size_t level);

protected:
	virtual void do_run();
	virtual void do_stop();

	void CheckCommands();
	void CheckTrackIsEnd();

public:
	typedef std::set<musicplayer_tracker_inf_t *> musicplayer_tracker_list_t;

private:
	string_v	m_vsTrackList;

	size_t		m_nCurrentTrack;
	EMPCOMMAND	m_empcState;

	event_t m_wait_event;

	musicplayer_tracker_list_t m_trackers;
};

#endif // __H_MUSICPLAYER__
