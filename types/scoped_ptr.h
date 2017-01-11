#ifndef __SCOPED_PTR_H_INCLUDED
#define __SCOPED_PTR_H_INCLUDED

#include "asserts.h"

template<typename T>
class scoped_ptr_t
{
public:
	scoped_ptr_t(T *pointer)
		: m_pointer(pointer)
		, m_bAttached(true) 
	{
		ASSERT(m_pointer != NULL);
	}

	~scoped_ptr_t()
	{
		if (m_bAttached)
		{
			delete m_pointer;
		}
	}

	T *detach()
	{
        ASSERT(m_bAttached);
        m_bAttached = false;

		return m_pointer;
	}

	T &operator*()
	{
		return *m_pointer;
	}

	const T &operator*() const
	{
		return *m_pointer;
	}

	T *operator->()
	{
		return m_pointer;
	}

	const T *operator->() const
	{
		return m_pointer;
	}

	T *get()
	{
		return m_pointer;
	}

	const T *get() const
	{
		return m_pointer;
	}

private:
	scoped_ptr_t<T>(const scoped_ptr_t<T> &other);
	scoped_ptr_t<T> &operator=(const scoped_ptr_t<T> &other);

	T *m_pointer;
	bool m_bAttached;
};

#endif // __SCOPED_PTR_H_INCLUDED
