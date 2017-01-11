#include "StdAfx.h"
#include "threading.h"

#include "asserts.h"
#include "platform.h"

#ifdef LINUX

#include <pthread.h>
#include <errno.h>


/////////////////////////////////////////////////////////////
// mutex_t implementation

class mutex_private_t
{
	friend class mutex_t;

	pthread_mutex_t m_mutex;
};

mutex_t::mutex_t()
	: m_private(new mutex_private_t())
{
	pthread_mutexattr_t m_mutex_attr;

	int rc = pthread_mutexattr_init(&m_mutex_attr);
	ASSERT(rc == 0);

	rc = pthread_mutexattr_settype(&m_mutex_attr, PTHREAD_MUTEX_RECURSIVE);
	ASSERT(rc == 0);

	rc = pthread_mutex_init(&(m_private->m_mutex), &m_mutex_attr);
	ASSERT(rc == 0);
}

mutex_t::~mutex_t()
{
	if (m_private)
	{
		int rc = pthread_mutex_destroy(&(m_private->m_mutex));
		ASSERT(rc == 0);

		delete m_private;
		m_private = NULL;
	}
}


void mutex_t::lock()
{
	int rc = pthread_mutex_lock(&(m_private->m_mutex));
	ASSERT(rc == 0);
}

void mutex_t::unlock()
{
	int rc = pthread_mutex_unlock(&(m_private->m_mutex));
	ASSERT(rc == 0);
}



/////////////////////////////////////////////////////////////
// mutex_t implementation
// Copied from http://qstuff.blogspot.com/2005/05/events-posix-version.html

class event_private_t
{
	friend class event_t;

	bool m_bAutoReset; // Constant

	pthread_mutex_t m_mutex; // Mutex
	pthread_cond_t m_cond; // Condition variable

	// Protected by m_mutex
	bool m_bSet; // Set or clear

	unsigned int m_nWaitingThreads; // Number of threads waiting on the event

	// As the name implies, this must be called inside the mutex
	// Does the wait. The parameter specifies when the thread should wake up, should the event not get set before then. If this is NULL, the thread will wait indefinitely on the event. Returns whether the event got set (if not, the timeout must have expired).
	bool inner_wait(const timespec *abstime)
	{
		if (!m_bSet)
		{
			int nRetVal;

			m_nWaitingThreads++;

			do
			{

				// Do the wait, either timed or indefinite
				if (abstime)
					nRetVal = pthread_cond_timedwait(&m_cond, &m_mutex, abstime);
				else
					nRetVal = pthread_cond_wait(&m_cond, &m_mutex);

				ASSERT(nRetVal == 0 || nRetVal == ETIMEDOUT);
			} while (!m_bSet && nRetVal != ETIMEDOUT); // Loop until it gets set or the timeout expires

			m_nWaitingThreads--;
		}

		// Did the event get set?
		bool bSuccess = m_bSet;
		// If the event is set and it's an auto-reset event, reset it now that we're awake
		if (m_bSet && m_bAutoReset)
			m_bSet = false;

		return bSuccess;
	}


};

event_t::event_t(bool manual_reset, bool is_set)
{
	m_private = new event_private_t();
	bool bAutoReset = !manual_reset;

	int res = pthread_mutex_init(&(m_private->m_mutex), NULL);
	ASSERT(res == 0);
	res = pthread_cond_init(&(m_private->m_cond), NULL);
	ASSERT(res == 0);

	m_private->m_bAutoReset = bAutoReset;
	m_private->m_bSet = is_set;

	m_private->m_nWaitingThreads = 0;
}

event_t::~event_t()
{
	if (m_private)
	{
		pthread_cond_destroy(&(m_private->m_cond));
		pthread_mutex_destroy(&(m_private->m_mutex));
		delete m_private;
		m_private = NULL;
	}
}


void event_t::set() const
{
	pthread_mutex_lock(&(m_private->m_mutex));

	if (!m_private->m_bSet) // If it's already set, do nothing
	{
		m_private->m_bSet = true; // Set the event

		// Check if there are any waiters, and release them appropriately
		if (m_private->m_nWaitingThreads)
		{
			if (m_private->m_bAutoReset)
				pthread_cond_signal(&(m_private->m_cond)); // Release one thread
			else
				pthread_cond_broadcast(&(m_private->m_cond)); // Release all threads
		}
	}

	pthread_mutex_unlock(&(m_private->m_mutex));
}

void event_t::reset() const
{
	pthread_mutex_lock(&(m_private->m_mutex));

	m_private->m_bSet = false; // Ding

	pthread_mutex_unlock(&(m_private->m_mutex));
}


bool event_t::wait(uint ms) const
{
	// Calculate the time to wake based on the time to sleep. I hope I understand how this is supposed to work on POSIX.
	timespec now, timeout, later;

	clock_gettime(CLOCK_REALTIME, &now);

	timeout.tv_sec = ms / 1000; // Seconds
	timeout.tv_nsec = (ms % 1000) * 1000000L; // Nanoseconds

	later.tv_sec = now.tv_sec + timeout.tv_sec;
	later.tv_nsec = now.tv_nsec + timeout.tv_nsec;
	if (later.tv_nsec >= 1000000000L)
	{
		later.tv_nsec -= 1000000000L;
		later.tv_sec++;
	}

	pthread_mutex_lock(&(m_private->m_mutex));

	bool bSuccess = m_private->inner_wait(&later);

	pthread_mutex_unlock(&(m_private->m_mutex));

	return bSuccess;
}

void event_t::wait() const
{
	pthread_mutex_lock(&(m_private->m_mutex));

	m_private->inner_wait(NULL);

	pthread_mutex_unlock(&(m_private->m_mutex));
}


/////////////////////////////////////////////////////////////
// thread_base_t implementation

void* thread_base_proc(void *self)
{
	thread_proc(self);
	return 0;
}

class thread_private_t
{
	friend class thread_base_t;

	pthread_t m_thread;
	volatile long m_join_counter;

	thread_private_t() : m_join_counter(0) { }
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

thread_base_t::~thread_base_t()
{
	ASSERT(!is_alive());

	if (m_private)
	{
		delete m_private;
		m_private = NULL;
	}
}


bool thread_base_t::start()
{
	ASSERT(m_tsState == TS_NOT_STARTED);
	m_tsState = TS_STARTING;	
	
	int rc = -1;
	
	do
	{
		rc = pthread_create(&(m_private->m_thread), NULL, &thread_base_proc, this);
	}
	while ((rc != 0) && (errno == EAGAIN));

	ASSERT(rc == 0);
	return true;
}


void thread_base_t::wait(unsigned long dwMilliseconds /*= INFINITE*/) const
{
	if (atomic_add(m_private->m_join_counter, 1) > 0)
	{
		return;
	}

	if (m_tsState == TS_NOT_STARTED)
	{
		return;
	}

	int res = pthread_join(m_private->m_thread, NULL);
	if (res != 0)
	{
		ASSERT(errno != EDEADLK);
	}
}

void thread_base_t::detach()
{
	int res = pthread_detach(m_private->m_thread);
	ASSERT(res == 0);
	m_bDetached = true;
}


long atomic_add (volatile long& value, long increment_)
{
#ifdef __GNUG__
	return __sync_fetch_and_add(&value, increment_);
#else
#	error NOT DEFINED FOR THIS COMPILER
#endif
}

uint64 atomic_add (volatile size_t& value, size_t increment_)
{
#ifdef __GNUG__
	return __sync_fetch_and_add(&value, increment_);
#else
#	error NOT DEFINED FOR THIS COMPILER
#endif
}


size_t get_current_thread_id()
{
	pthread_t thread_id = pthread_self();
	// XXX: will break on systems where pthread_t is not defined as numerical
	return (size_t)thread_id;
}


#endif // #ifdef LINUX
