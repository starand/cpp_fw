#ifndef __VOLUME_TRACKER_H__INCLUDED
#define __VOLUME_TRACKER_H__INCLUDED

#include <threading.h>
#include <volume_mixer.h>

#include <set>


class volume_tracker_inf_t
{
public:
	virtual ~volume_tracker_inf_t() { }

	virtual void on_volume_changed(size_t level) = 0;
};

//////////////////////////////////////////////////

class volume_tracker_t : public thread_base_t
{
public:
	volume_tracker_t();
	~volume_tracker_t();

public: // thread_base_t implementation
	virtual void do_run();
	virtual void do_stop();

public:
	void add_tracker(volume_tracker_inf_t *tracker);
	void remove_tracker(volume_tracker_inf_t *tracker);

public:
	size_t get_level() const;
	void set_level(size_t level);

private:
	void check_level();

private:
	typedef std::set<volume_tracker_inf_t *> trackers_list_t;

private:
	volume_mixer_t m_volume_mixer;

	event_t m_wait_event;
	size_t m_current_level;

	trackers_list_t m_volume_trackers;
};

#endif // __VOLUME_TRACKER_H__INCLUDED
