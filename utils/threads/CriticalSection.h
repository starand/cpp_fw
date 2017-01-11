#ifndef __H_CRITICALSECTION__
#define __H_CRITICALSECTION__


class CCriticalSection
{
public:
	CCriticalSection() : pCSObject(NULL)
	{
		pCSObject = new (nothrow) CRITICAL_SECTION;
		ASSERTE(pCSObject);

		::InitializeCriticalSection((CRITICAL_SECTION *)pCSObject);
	}

	~CCriticalSection()
	{
		::DeleteCriticalSection((CRITICAL_SECTION *)pCSObject);
	}

public:
	void Lock()
	{
		::EnterCriticalSection((CRITICAL_SECTION *)pCSObject);
	}

	void Unlock()
	{
		::LeaveCriticalSection((CRITICAL_SECTION *)pCSObject);
	}

	bool TryLock()
	{
		return ( ::TryEnterCriticalSection((CRITICAL_SECTION *)pCSObject) != 0 );
	}

private:
	void* pCSObject;
};

#endif // __H_CRITICALSECTION__
