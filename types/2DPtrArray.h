#ifndef __2DPTRARRAY_H_INCLUDED
#define __2DPTRARRAY_H_INCLUDED

//////////////////////////////////////////////////
// C2DPtrArray declaration

template<typename TElem>
class C2DPtrArray
{
public:
	C2DPtrArray(): 
		m_ppElements(NULL), 
		m_siElementsCount(0),
		m_siTotalElemesCount(0)
	{
	}

	C2DPtrArray(sizeint siArraySize): 
		m_ppElements(NULL), 
		m_siElementsCount(siArraySize),
		m_siTotalElemesCount(0)
	{
		AllocateMemory(siArraySize);
	}

	~C2DPtrArray<TElem>()
	{
		FreeMemory();
	}

private:
	typedef TElem* TElemPtr;
	typedef const TElem* TConstElemPtr;

public:
	TElemPtr &operator[](sizeint siIndex) 
	{
		ASSERTE(siIndex < m_siElementsCount);
		return m_ppElements[siIndex];
	}

	TElemPtr &operator[](sizeint siIndex) const
	{
		ASSERTE(siIndex < m_siElementsCount);
		return m_ppElements[siIndex];
	}

	void SetElement(sizeint siIndex, TConstElemPtr ppElements, sizeint siElementsCount)
	{
		ASSERTE(siIndex < m_siElementsCount);

		AllocateElementMemory(siIndex, siElementsCount);

		sizeint siMemorySize = siElementsCount * sizeof(TElem);
		memcpy(m_ppElements[siIndex], ppElements, siMemorySize);
	}

	TElemPtr* GetDataPtr() const { return m_ppElements; }
	sizeint GetDataSize() const { return m_siElementsCount; }

public:
	operator bool() const { return m_ppElements != NULL; }

	void operator=(C2DPtrArray &paOther)
	{
		m_ppElements = paOther.m_ppElements;
		m_siElementsCount = paOther.m_siElementsCount;
		m_siTotalElemesCount = paOther.m_siTotalElemesCount;

		paOther.m_ppElements = NULL;
		paOther.m_siElementsCount = 0;
		paOther.m_siTotalElemesCount = 0;
	}

	C2DPtrArray<TElem>(C2DPtrArray<TElem> &paOther)
	{
		*this = paOther;
	}

public:
	sizeint GetTotalElemsCount() const { return m_siTotalElemesCount; }

private:
	void AllocateMemory(sizeint siArraySize)
	{
		ASSERTE(m_ppElements == NULL && m_siElementsCount);

		TElemPtr *ppElements = new (nothrow) TElemPtr[m_siElementsCount];
		ASSERTE(ppElements);

		m_ppElements = ppElements;
		m_siElementsCount = siArraySize;

		sizeint siMemorySize = m_siElementsCount * sizeof(TElemPtr);
		memset(m_ppElements, 0, siMemorySize);
	}

	void FreeMemory()
	{
		if (m_ppElements)
		{
			for (sizeint siIndex = 0; siIndex < m_siElementsCount; ++siIndex)
			{
				FreeElementMemory(siIndex);
			}

			delete [] m_ppElements;
			m_ppElements = NULL;
		}
	}


	void AllocateElementMemory(sizeint siIndex, sizeint siElementsCount)
	{
		ASSERTE(siIndex < m_siElementsCount);
		ASSERTE(m_ppElements[siIndex] == NULL);

		TElemPtr ppElements = new (nothrow) TElem[siElementsCount];
		ASSERTE(ppElements);

		m_ppElements[siIndex] = ppElements;
		m_siTotalElemesCount += siElementsCount;
	}

	void FreeElementMemory(sizeint siIndex)
	{
		ASSERTE(siIndex < m_siElementsCount);
		
		if (m_ppElements[siIndex])
		{
			delete [] m_ppElements[siIndex];
			m_ppElements[siIndex] = NULL;
		}
	}

private:
	TElemPtr	*m_ppElements;

	sizeint		m_siElementsCount;
	sizeint		m_siTotalElemesCount;
};

#endif // __2DPTRARRAY_H_INCLUDED
