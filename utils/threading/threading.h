#ifndef __THREADING_H_INCLUDED
#define __THREADING_H_INCLUDED

#include "types.h"


///////////////////////////////////////////////////
// mutex_t declaration

class mutex_private_t;

class mutex_t
{
public:
	mutex_t();
	~mutex_t();

	void lock();
	void unlock();

private:
	mutex_private_t *m_private;
};


class mutex_locker_t
{
public:
	mutex_locker_t(mutex_t& mutex) : m_mutex(mutex)
	{
		m_mutex.lock();
	}

	~mutex_locker_t()
	{
		m_mutex.unlock();
	}

private:
	mutex_t& m_mutex;
};


///////////////////////////////////////////////////
// event_t declaration

class event_private_t;

class event_t
{
public:
	event_t(bool manual_reset = true, bool is_set = false);
	~event_t();

	void set() const;
	void reset() const;

	void wait() const;
	bool wait(uint ms) const;

private:
	event_private_t *m_private;
};


///////////////////////////////////////////////////
// thread_base_t declaration

#ifndef INFINITE
#       define INFINITE 0xFFFFFFFF
#endif // INFINITE

#ifdef sleep
#   undef sleep
#endif

class thread_private_t;

class thread_base_t
{
public:
	thread_base_t(bool bAutoDelete = false);
	virtual ~thread_base_t();

public:
	virtual void do_run() = 0;
	virtual void do_stop() { }

public:
	void run();

	bool start();
	void stop();

	void wait(unsigned long dwMilliseconds = INFINITE) const;
	void detach();

	void sleep() const;
	void sleep(unsigned long ulMilliseconds) const;

	bool is_auto_delete() const { return m_bAutoDelete; }
	bool is_alive() const { return m_tsState == TS_STARTING || m_tsState == TS_WORKING; }
	bool is_stopping() const { return m_bStopping; }

public:
	enum thread_state_t { TS_NOT_STARTED, TS_STARTING, TS_WORKING, TS_STOPPED };

private:
	bool m_bAutoDelete;
	bool m_bDetached;

	thread_private_t *m_private;
	thread_state_t  m_tsState;
	bool m_bStopping;
	event_t m_eStopEvent;
};


void thread_proc(void *);

long atomic_add(volatile long& value, long increment_);
uint64 atomic_add(volatile size_t& value, size_t increment_);

size_t get_current_thread_id();

#endif // __THREADING_H_INCLUDED
