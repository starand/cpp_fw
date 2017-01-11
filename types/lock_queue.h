#ifndef __LOCK_QUEUE_H_INCLUDED
#define __LOCK_QUEUE_H_INCLUDED

#include <threading.h>


template<class T> class lock_queue_t
{
	struct node_t
	{
		node_t() : next(NULL) { }

		T data;
		node_t* next;
	};

public:
	lock_queue_t()
		: m_start(NULL)
		, m_end(NULL)
		, m_count(0)
		, m_mutex()
		, m_event(false, false)
	{
	}

	~lock_queue_t()
	{
		cleanup();
	}

	void push(T& data)
	{
		node_t* node = new node_t;
		node->data = data;

		mutex_locker_t lock(m_mutex);
		if (m_start != NULL&&  m_end != NULL)
		{
			m_end->next = node;
			m_end = node;
		}
		else
		{
			m_start = m_end = node;
		}

		if (!m_count++)
		{
			m_event.set();
		}
	}

	T pop()
	{
		m_mutex.lock();
		while (empty())
		{
			m_mutex.unlock();
			m_event.wait();
			m_mutex.lock();
		}

		ASSERT(m_start != NULL);
		T data = m_start->data;

		delete_front();
		m_mutex.unlock();

		return data;
	}

	bool empty() const
	{
		return m_count == 0;
	}

	size_t size() const
	{
		return m_count;
	}

	void cleanup()
	{
		mutex_locker_t lock(m_mutex);
		while (m_start)
		{
			delete_front();
		}
	}

private:
	node_t* delete_front() // thread unsafe; returns next node after deleted
	{
		ASSERT(m_start != NULL);
		node_t *node = m_start;

		if (m_start == m_end)
		{
			m_start = m_end = NULL;
		}
		else
		{
			m_start = m_start->next;
		}

		delete node;
		--m_count;
		return m_start;
	}

private:
	node_t* m_start;
	node_t* m_end;
	size_t m_count;

	mutex_t m_mutex;
	event_t m_event;
};

#endif // __LOCK_QUEUE_H_INCLUDED
