#ifndef __VIRTUALMEMORY_H_INCLUDED
#define __VIRTUALMEMORY_H_INCLUDED

//////////////////////////////////////////////////
// CVirtualMemory declaration

template<typename TElemType>
class CVirtualMemory
{
public:
	CVirtualMemory(sizeint nElementsCount):
		m_nElementsCount(nElementsCount),
		m_pvMemoryPtr(NULL)
	{
		m_pvMemoryPtr = VirtualAlloc(NULL, nElementsCount * sizeof(TElemType), MEM_COMMIT, PAGE_READWRITE);
	}

	virtual ~CVirtualMemory() 
	{
		if (m_pvMemoryPtr)
		{
			VirtualFree(m_pvMemoryPtr, 0, MEM_RELEASE);
		}
	}

public:
	bool GetAllocationFailed() const { return m_pvMemoryPtr == NULL; }

	operator TElemType*() const { return static_cast<TElemType *>(m_pvMemoryPtr); }

private:
	sizeint		m_nElementsCount;
	void		*m_pvMemoryPtr;
};

#endif // __VIRTUALMEMORY_H_INCLUDED
