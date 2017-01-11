#ifndef __H_PRIORITYQUEUE__
#define __H_PRIORITYQUEUE__

template<typename T>
class CPriorityQueue
{
	struct CNode
	{
		CNode( const T& newItem, CNode* pNextNode = NULL ) : tData(newItem), pNext(pNextNode) { }

		CNode*	pNext;
		T		tData;
	};

public:
	CPriorityQueue() : m_pHead(NULL) { }
	~CPriorityQueue()
	{
		while( m_pHead ) {
			CNode* pCurrent = m_pHead;
			m_pHead = m_pHead->pNext;
			delete pCurrent;
		}
	}

	void push( const T& tItem )
	{
		if( !m_pHead )
		{
			m_pHead = new CNode( tItem );
		}
		else if( tItem < m_pHead->tData )
		{
			m_pHead = new CNode( tItem, m_pHead );
		}
		else
		{
			CNode* pCurrent = m_pHead;
			while( pCurrent->pNext && tItem < pCurrent->pNext->tData ) pCurrent = pCurrent->pNext;
			pCurrent->pNext = new CNode( tItem, pCurrent->pNext );
		}
#ifdef DEBUG_PRIORITYQUEUE
		_debug();
#endif
	}

	T pop()
	{
		_ASSERT( m_pHead );
		if( !m_pHead ) throw out_of_range( "empty queue" );

		CNode* pCurrent = m_pHead;
		m_pHead = m_pHead->pNext;

		T tResItem = pCurrent->tData;
		delete pCurrent;
		return tResItem;
#ifdef DEBUG_PRIORITYQUEUE
		_debug();
#endif
	}

	T* begin()
	{
		return ( m_pHead ? &m_pHead->tData : NULL );
	}

	bool empty() { return NULL == m_pHead; }

protected:
#ifdef DEBUG_PRIORITYQUEUE
	virtual void _debug()
	{
		CNode* pCurrent = m_pHead;
		while( pCurrent ) {
			cout << hex << "0x" << pCurrent << " - " << dec << pCurrent->tData << "  ";
			pCurrent = pCurrent->pNext;
		}
		cout << endl;
	}
#endif

private:
	CNode* m_pHead;
};

#endif // __H_PRIORITYQUEUE__
