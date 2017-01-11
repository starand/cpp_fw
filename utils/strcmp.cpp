#include "stdafx.h"
#include "strcmp.h"
#include "macroes.h"
#include "safebuffer.h"

//////////////////////////////////////////////////////
// CStrComparator implementation

CStrComparator::CStrComparator( const char* szValue, bool bCaseSnsitive /*= true*/, size_t nComparePos /*= 0*/, size_t nLength /*= 0*/ )
	: m_szData( szValue ), m_bCaseSensitive(bCaseSnsitive), m_nComparePos(nComparePos), m_nLength(nLength)
{ }

bool CStrComparator::operator<( const CStrComparator& rOp ) const
{
	if( m_bCaseSensitive )
	{
		if( m_nLength ) return strncmp( m_szData+m_nComparePos, rOp.m_szData+m_nComparePos, m_nLength ) < 0;
		else return strcmp( m_szData+m_nComparePos, rOp.m_szData+m_nComparePos ) < 0;
	}
	else 
	{
		if( m_nLength ) return _strnicmp( m_szData+m_nComparePos, rOp.m_szData+m_nComparePos, m_nLength ) < 0;
		else return _stricmp( m_szData+m_nComparePos, rOp.m_szData+m_nComparePos ) < 0;
	}
}


//////////////////////////////////////////////////////
// CSortedStringMap implementation

CSortedStringMap::CSortedStringMap( char cSeparator /*= '\n'*/, bool bIgnoreCase /*= false*/ ) 
	: m_cSeparator(cSeparator), m_bIgnoreCase(bIgnoreCase), m_nComparePos(), m_nLength()
{
}

CSortedStringMap::CSortedStringMap( bool bIgnoreCase, char cSeparator /*= '\n'*/ )
	: m_cSeparator(cSeparator), m_bIgnoreCase(bIgnoreCase), m_nComparePos(), m_nLength()
{
}

CSortedStringMap::CSortedStringMap( size_t nComparePos, size_t nLength, char cSeparator /*= '\n'*/, bool bIgnoreCase /*= false*/ )
	: m_nComparePos(nComparePos), m_nLength(nLength), m_cSeparator(cSeparator), m_bIgnoreCase(bIgnoreCase)
{
}

void CSortedStringMap::LoadFromBuffer( char* szBuffer )
{
	clear();
	char* szCurrent = szBuffer;
	while( szCurrent && *szCurrent )
	{
		char* szNext = strchr( szCurrent, m_cSeparator );
		if( szNext ) { *szNext = 0; ++szNext; }

		insert( CSortedStringMap::value_type( CStrComparator(szCurrent, !m_bIgnoreCase, m_nComparePos, m_nLength), szCurrent) );
		szCurrent = szNext;
		
	}
}

bool CSortedStringMap::LoadFromFile( const char* szFileName, char* szBuffer, size_t nBufferSize )
{
	bool bResult = false;

	FILE* fp = NULL;
	do
	{
		if( !szFileName || *szFileName == 0 || fopen_s(&fp, szFileName, "r")  || fseek(fp, 0, SEEK_END) ) break;
		
		uint64 nFileSize = ftell( fp );
		if( nFileSize >= (nBufferSize - 1) || nFileSize == -1L ) break;
		if( fseek(fp, 0, SEEK_SET) ) break;

		nFileSize =  fread( szBuffer, 1, nBufferSize, fp );
		if( ferror(fp) ) break;

		szBuffer[ nFileSize ] = 0;
		LoadFromBuffer( szBuffer );
		bResult = true;
	}
	while( false );
	CLOSE_FILE( fp );

	return bResult;
}

void CSortedStringMap::Show( const char* szLineSeparator /*= "\n"*/ ) const
{
	for( CSortedStringMap_Parent::const_iterator itMap = begin(); itMap != end(); itMap++ )
	{
		if( itMap->second ) cout << itMap->second << ( szLineSeparator ? szLineSeparator : "\n" );
	}
}

bool CSortedStringMap::SaveToFile( const char* szFileName, const char* szLineSeparator /*= "\n"*/ ) const
{
	bool bResult = false;

	FILE* fp = NULL;
	do
	{
		bool bError = false;
		if( !szFileName || *szFileName == 0 || fopen_s(&fp, szFileName, "w") ) break;
		for( CSortedStringMap_Parent::const_iterator itMap = begin(); itMap != end(); itMap++ )
		{
			if( !itMap->second ) continue;

			fprintf( fp, "%s%s", itMap->second, szLineSeparator ? szLineSeparator : "\n" );
			if( ferror(fp) )
			{
				bError = true;
				break;
			}
		}

		bResult = !bError;
	}
	while( false );
	CLOSE_FILE( fp );

	return bResult;
}

