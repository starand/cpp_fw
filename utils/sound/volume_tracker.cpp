#include "StdAfx.h"
#include "volume_tracker.h"


#define CHECK_TIMEOUT 200


//////////////////////////////////////////////////
// CMixerVolume implementation

volume_tracker_t::volume_tracker_t()
	: m_volume_mixer()
	, m_wait_event(false, false)
	, m_current_level(0)
	, m_volume_trackers()
{
}

volume_tracker_t::~volume_tracker_t()
{
}


/*virtual */
void volume_tracker_t::do_run()
{
	while (true)
	{
		m_wait_event.wait(CHECK_TIMEOUT);
		if (is_stopping())
		{
			break;
		}

		check_level();
	}
}

/*virtual */
void volume_tracker_t::do_stop()
{
	m_wait_event.set();
}


void volume_tracker_t::add_tracker(volume_tracker_inf_t *tracker)
{
	m_volume_trackers.insert(tracker);
}

void volume_tracker_t::remove_tracker(volume_tracker_inf_t *tracker)
{
	m_volume_trackers.erase(tracker);
}


// return level in range 0 .. 1000
size_t volume_tracker_t::get_level() const
{
	return m_current_level;
}

void volume_tracker_t::set_level(size_t level)
{
	float fLevel = (level % 1000) / 1000.0;
	m_volume_mixer.set_volume_level(fLevel, true);
}


void volume_tracker_t::check_level()
{
	float fLevel = 0;
	if (m_volume_mixer.get_volume_level(fLevel, true))
	{
		size_t level = (size_t)(fLevel * 1000);

		if (level != m_current_level)
		{
			m_current_level = level;

			trackers_list_t::const_iterator iter = m_volume_trackers.begin();
			for (; iter != m_volume_trackers.end(); ++iter)
			{
				(*iter)->on_volume_changed(level);
			}
		}
	}
}
