#ifndef __H_ALGORITHM__
#define __H_ALGORITHM__

#ifndef min
#	define min( a, b ) ((a) > (b) ? (b) : (a))
#endif

namespace Algorithm
{

	template<class T> void swap( T& a, T& b )
	{
		T tmp = a; a = b; b = tmp;
	}

	template<typename T>
	bool CheckIfValuesExists( const vector<T>& vtList, const T& tValue )
	{
		bool bResult = false;

		size_t nListSize = vtList.size();
		for( size_t idx = 0; idx < nListSize; ++idx )
		{
			if( vtList[idx] == tValue )
			{
				bResult = true;
				break;
			}
		}

		return bResult;
	}

	template<typename T>
	size_t GetValuePos( const T* tArray, size_t nArraySize, const T& tValue )
	{
		for( size_t idx = 0; idx < nArraySize; ++idx )
			if( tArray[idx] == tValue ) return idx;

		return string::npos;
	}
}

#endif // __H_ALGORITHM__
