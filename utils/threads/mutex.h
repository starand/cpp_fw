#ifndef __H_MUTEX__
#define __H_MUTEX__

class CMutex
{
public:
	CMutex();
	~CMutex();

	void Lock();
	void Unlock();
	bool Locked() { return m_bLocked; }

private:
	HANDLE	m_hMutex;
	size_t	m_dwWaitResult;
	bool	m_bLocked;
};

#endif // __H_MUTEX__
