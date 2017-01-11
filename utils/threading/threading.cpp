#include "StdAfx.h"
#include "threading.h"

#include "asserts.h"


//////////////////////////////////////////////
// thread_base_t implementation

void thread_proc(void *self)
{
	thread_base_t *thread = (thread_base_t *)self;
	thread->run();

	if (thread->is_auto_delete())
	{
		thread->detach();
		delete thread;
	}
}

void thread_base_t::run()
{
	ASSERTE(m_tsState == TS_STARTING);
	m_tsState = TS_WORKING;
	do_run();
	m_tsState = TS_STOPPED;
}


void thread_base_t::stop()
{
	m_bStopping = true;
	m_eStopEvent.set();
	do_stop();
}


void thread_base_t::sleep() const
{
	m_eStopEvent.wait();
}

void thread_base_t::sleep(unsigned long ulMilliseconds) const
{
	m_eStopEvent.wait(ulMilliseconds);
}
