#ifndef __OPERATION_ACTIVITY_H_INCLUDED
#define __OPERATION_ACTIVITY_H_INCLUDED

#include <time.h>

class operation_activity_t
{
public:
	void set_busy_now()
	{
		m_bBusy = true;
		m_tLastTimestamp = time(NULL);
	}

	void set_idle_now()
	{
		m_bBusy = false;
		m_tLastTimestamp = time(NULL);
	}

	const double get_idle_time() const
	{
		return m_bBusy ? 0.0 : difftime(time(NULL), m_tLastTimestamp);
	}

	const double get_busy_time() const
	{
		return m_bBusy ? difftime(time(NULL), m_tLastTimestamp) : 0.0;
	}

private:
	volatile bool	m_bBusy;
	volatile time_t	m_tLastTimestamp;
};

#endif // __OPERATION_ACTIVITY_H_INCLUDED
