#ifndef __H_QUEUE__
#define __H_QUEUE__

#include "criticalsection.h"
#include "autolock.h"

template<class T> class CQueue
{
protected:
	struct CNode 
	{ 
		CNode() : next(NULL) { }
		T data; CNode* next; 
	};

public:
	CQueue() : m_pStart(NULL), m_pEnd(NULL), m_nCount(0) { }

	void push( T newElem )
	{
		ASSERTE(!(!m_pStart && m_pStart != m_pEnd));

		CNode* pNewNode = new CNode;
		pNewNode->data = newElem;

		if( m_pStart && m_pEnd )
		{
			m_pEnd->next = pNewNode;
			m_pEnd = pNewNode;
		}
		else
		{
			m_pStart = m_pEnd = pNewNode;
		}

		++m_nCount;
		LOG_TRACE2("Added element. Count : %u", m_nCount);
	}

	T pop()
	{
		ASSERTE(m_pStart);
		CNode* pResult = m_pStart;

		if( m_pStart == m_pEnd ) m_pStart = m_pEnd = NULL;
		else m_pStart = m_pStart->next;

		T tResult = pResult->data;
		delete pResult;
		--m_nCount;

		LOG_TRACE2("Removed element. Count : %u", m_nCount);
		return tResult;		
	}

	bool empty(){ if (m_pStart == NULL) { ASSERTE(!m_nCount); } return m_pStart == NULL; }
	size_t size() { return m_nCount; }

private:
	CNode* m_pStart;
	CNode* m_pEnd;

	size_t m_nCount;
};

template<class T> class CLockQueue : public CQueue<T>
{
public:
	void push( T newElem )
	{
		CAutoLock<> lock( m_cssMutex );
		CQueue<T>::push( newElem );
	}

	T pop()
	{
		CAutoLock<> lock( m_cssMutex );
		return CQueue<T>::pop();
	}

private:
	CCriticalSection m_cssMutex;

};

#endif // __H_QUEUE__
