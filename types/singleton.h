#ifndef __H_SINGLETON__
#define __H_SINGLETON__

template<typename T>
class CSingleton
{
public:
	static T* Instance()
	{
		do 
		{
			if( m_pPointee ) break;

			while( m_bLock );

			m_bLock = true;
			if( !m_pPointee )
			{
				m_pPointee = new T;
			}
			m_bLock = false;
		} 
		while(false);
		return m_pPointee;
	}

protected:
	CSingleton() {}
	~CSingleton() {}
	CSingleton(CSingleton& obj);
	CSingleton operator=(CSingleton& obj);

	void FreeObject()
	{
		if( m_pPointee )
		{
			delete m_pPointee;
			m_pPointee = NULL;
		}
	}

private:
	static T*	m_pPointee;
	static bool	m_bLock;
};

template<typename T> T* CSingleton<T>::m_pPointee = NULL;
template<typename T> bool CSingleton<T>::m_bLock = false;

#endif // __H_SINGLETON__
