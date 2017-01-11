#ifndef __H_STR_COMPARE__
#define __H_STR_COMPARE__

#include "types.h"

class CStrComparator
{
public:
	CStrComparator( const char* szValue, bool bCaseSnsitive = true, size_t nComparePos = 0, size_t nLength = 0 );

	bool operator<( const CStrComparator& rOp ) const;

private:
	const char* m_szData;
	bool m_bCaseSensitive;

	size_t m_nComparePos;
	size_t m_nLength;
};


typedef multimap< CStrComparator, const char* > CSortedStringMap_Parent;

class CSortedStringMap : public CSortedStringMap_Parent
{
public:
	CSortedStringMap( char cSeparator = '\n', bool bIgnoreCase = false );
	CSortedStringMap( bool bIgnoreCase, char cSeparator = '\n' );
	CSortedStringMap( size_t nComparePos, size_t nLength, char cSeparator = '\n', bool bIgnoreCase = false );

	void LoadFromBuffer( char* szBuffer );
	bool LoadFromFile( const char* szFileName, char* szBuffer, size_t nBufferSize );

	void Show( const char* szLineSeparator = "\n" ) const;
	bool SaveToFile( const char* szFileName, const char* szLineSeparator = "\n" ) const;

private:
	char m_cSeparator;
	bool m_bIgnoreCase;

	size_t m_nComparePos;
	size_t m_nLength;
};


#endif // __H_STR_COMPARE__
