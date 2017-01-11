#ifndef __H_AUTOHANDLE__
#define __H_AUTOHANDLE__

class CAutoHandle
{
public:
	CAutoHandle(HANDLE hHandle = NULL) : m_Handle(hHandle) { }
	~CAutoHandle() 
	{ 
		if( m_Handle ) 
		{
			CloseHandle( m_Handle );
			m_Handle = NULL;
		}
	}

	HANDLE GetHandle() { return m_Handle; }

protected:
	HANDLE	m_Handle;
};

#endif // __H_AUTOHANDLE__
