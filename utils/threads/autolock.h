#ifndef __H_AUTOLOCK__
#define __H_AUTOLOCK__

template<typename T = CCriticalSection>
class CAutoLock
{
public:
	CAutoLock(T& lockObj, bool bLock = true);
	~CAutoLock();

	void Lock();
	void Unlock();
	
private:
	T&	m_pLockRef;
};

/////////////////////////////////////////////////////////
// CAutoLock implementation

template<typename T>
CAutoLock<T>::CAutoLock(T& lockObj, bool bLock /* = true */)
	: m_pLockRef(lockObj)
{
	if( bLock ) m_pLockRef.Lock();
}

template<typename T>
CAutoLock<T>::~CAutoLock()
{
	m_pLockRef.Unlock();
}

template<typename T>
void CAutoLock<T>::Lock()
{
	return m_pLockRef.Lock();
}

template<typename T>
void CAutoLock<T>::Unlock()
{
	return m_pLockRef.Unlock();
}


#endif // __H_AUTOLOCK__
