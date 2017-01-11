#ifndef _SMART_POINTER_
#define _SMART_POINTER_

template<class T> class CAutoPtr
{
private:
	typedef	T* POINTER;

	POINTER		m_ptr;

public:
	CAutoPtr(POINTER ptr) : m_ptr(ptr) {}

	~CAutoPtr()
	{
		if(m_ptr != NULL)
		{
			delete m_ptr;
			m_ptr = NULL;
		}
	}

	POINTER get() { return m_ptr; }

	T& operator*() const { return *m_ptr; }
	POINTER operator->() const {return m_ptr; }

private:
	CAutoPtr();
	CAutoPtr(const CAutoPtr& ob);
	void operator=(const CAutoPtr& ob);
};

#endif // _SMART_POINTER_
