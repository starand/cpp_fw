#ifndef _SMART_POINTER_
#define _SMART_POINTER_

template<class T> class CSmartPtr
{
private:
	T*     m_ptr;
	size_t*     m_pCounter;

	size_t IncLinkCount()
	{
		if(m_pCounter) 
		{
			(*m_pCounter)++;
			return *m_pCounter;
		}
		return 0;
	}

	size_t DecLinkCount()
	{
		if(m_pCounter && *m_pCounter > 0) 
		{
			(*m_pCounter)--;
			return *m_pCounter;
		}
		return 0;
	}

public:
	CSmartPtr(T* ptr) : m_ptr(ptr), m_pCounter(NULL)
	{
		m_pCounter = new size_t;
		*m_pCounter = 0;
		IncLinkCount();
	}

	CSmartPtr(const CSmartPtr& ob) : m_ptr(ob.m_ptr), m_pCounter(NULL)
	{
		m_pCounter = ob.m_pCounter;
		IncLinkCount();
	}

	~CSmartPtr()
	{
		if(m_ptr)
		{
			if(DecLinkCount() == 0)
			{
				delete m_ptr;
				m_ptr = 0;

				if(m_pCounter)
				{
					delete m_pCounter;
					m_pCounter = NULL;
				}
			}
		}
	}

	T* get() { return m_ptr; }
	T& operator*() const { return *m_ptr; }

private:
	CSmartPtr();
	void operator=(const CSmartPtr& ob);
};

#endif // _SMART_POINTER_
