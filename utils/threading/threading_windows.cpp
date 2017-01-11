#include "StdAfx.h"
#include "threading.h"

#include "asserts.h"
#include "platform.h"

#ifdef WINDOWS

#include <windows.h>

/////////////////////////////////////////////////////////////
// mutex_t implementation

class mutex_private_t
{
	friend class mutex_t;

	CRITICAL_SECTION m_critical_section;
};

mutex_t::mutex_t()
	: m_private(new mutex_private_t())
{
	InitializeCriticalSection(&(m_private->m_critical_section));
}

mutex_t::~mutex_t()
{
	if (m_private)
	{
		DeleteCriticalSection(&(m_private->m_critical_section));
		delete m_private;
		m_private = NULL;
	}
}


void mutex_t::lock()
{
	EnterCriticalSection(&(m_private->m_critical_section));
}

void mutex_t::unlock()
{
	LeaveCriticalSection(&(m_private->m_critical_section));
}


/////////////////////////////////////////////////////////////
// event_t implementation

class event_private_t
{
	friend class event_t;

	HANDLE m_event;
};

event_t::event_t(bool manual_reset /*= true*/, bool is_set /*= false*/)
{
	m_private = new event_private_t();
	m_private->m_event = CreateEvent(NULL, manual_reset ? TRUE : FALSE, is_set ? TRUE : FALSE, NULL);
	ASSERT(m_private->m_event != NULL);
}

event_t::~event_t()
{
	if (m_private)
	{
		CloseHandle(m_private->m_event);
		delete m_private;
		m_private = NULL;
	}
}


void event_t::set() const
{
	BOOL res = SetEvent(m_private->m_event);
	ASSERT(res == TRUE);
}

void event_t::reset() const
{
	BOOL res = ResetEvent(m_private->m_event);
	ASSERT(res == TRUE);
}


void event_t::wait() const
{
	DWORD res = WaitForSingleObject(m_private->m_event, INFINITE);
	ASSERT(res == WAIT_OBJECT_0);
}

bool event_t::wait(uint ms) const
{
	int res = WaitForSingleObject(m_private->m_event, ms);
	return (res == WAIT_OBJECT_0);
}


/////////////////////////////////////////////////////////////
// thread_base_t implementation

DWORD __stdcall thread_base_proc(void *self)
{
	thread_proc(self);
	return 0;
}


class thread_private_t
{
	friend class thread_base_t;

	HANDLE m_thread;

	thread_private_t() : m_thread(NULL) { }
};


thread_base_t::thread_base_t(bool bAutoDelete /*= false*/):
	m_bAutoDelete(bAutoDelete),
	m_bDetached(false),
	m_private(new thread_private_t()),
	m_tsState(TS_NOT_STARTED),
	m_bStopping(false),
	m_eStopEvent(true, false)
{
}

/*virtual */
thread_base_t::~thread_base_t()
{
	ASSERT(!is_alive());

	if (m_private)
	{
		if (m_private->m_thread)
		{
			CloseHandle(m_private->m_thread);
			m_private->m_thread = NULL;
		}

		delete m_private;
		m_private = NULL;
	}
}


bool thread_base_t::start()
{
	ASSERT(m_private->m_thread == NULL);
	ASSERT(m_tsState == TS_NOT_STARTED);

	m_tsState = TS_STARTING;
	m_private->m_thread = CreateThread(NULL, 0, &thread_base_proc, this, 0, 0);
	ASSERT(NULL != m_private->m_thread);
	return NULL != m_private->m_thread;
}


void thread_base_t::wait(unsigned long dwMilliseconds /*= INFINITE*/) const
{
	DWORD dwResult = ::WaitForSingleObject(m_private->m_thread, dwMilliseconds);
	ASSERT(dwResult == 0);
}

void thread_base_t::detach()
{
	m_bDetached = true;
	// TODO: Implement detaching for Windows
}


long atomic_add (volatile long& value, long increment_)
{
	long old_value;
	old_value = InterlockedExchangeAdd (&value, increment_);
	return old_value;
}

uint64 atomic_add (volatile size_t& value, size_t increment_)
{
	uint64 old_value;
	volatile LONG64 *val_ptr = (volatile LONG64 *)&value;
	old_value = InterlockedExchangeAdd64 (val_ptr, increment_);
	return old_value;
}


size_t get_current_thread_id()
{
	return (size_t)GetCurrentThreadId();
}

#endif // #ifdef WINDOWS
