#include "stdafx.h"
#include "textfile.h"
#include "logerror.h"
#include "macroes.h"
#include "fileutils.h"
#include "consts.h"
#include "algorithm.h"


//////////////////////////////////////////////////
// CTextFile definition

CTextFile::CTextFile( const string& sFileName )
{
	LoadFromFile( sFileName );
}

bool CTextFile::SaveToFile( const string& sFileName ) const
{
	return FileUtils::PutFileStrings( sFileName.c_str(), *this );
}

bool CTextFile::LoadFromFile( const string& sFileName, bool bClear /*= true*/ )
{
	if( bClear ) (*this).clear();
	return FileUtils::GetFileStrings( sFileName.c_str(), *this );
}

void CTextFile::GetLinesWith( const string& sSubString, string_v& vsResult, bool bClean /*= true*/ )
{
	if( bClean ) vsResult.clear();
	for( auto& sLine : *this )
	{
		if( string::npos != sLine.find(sSubString) ) vsResult.push_back( sLine );
	}
}

void CTextFile::GetLinesWith( const string_v& vsSubStrList, string_v& vsResult, bool bClean /*= true*/ )
{
	if( bClean ) vsResult.clear();
	for( auto& sLine : *this )
	{
		for( auto& sCheckValue : vsSubStrList )
		{
			if( string::npos != sLine.find(sCheckValue) )
			{
				vsResult.push_back( sLine );
				break;
			}
		}
	}
}

void CTextFile::GetLinesWithout( const string& sSubString, string_v& vsResult, bool bClean /*= true*/ )
{
	if( bClean ) vsResult.clear();
	for( auto& sLine : *this )
	{
		if( string::npos == sLine.find(sSubString) ) vsResult.push_back( sLine );
	}
}

void CTextFile::ExcludeWith( const string& sSubString )
{
	CTextFile tfResult;
	for( auto& sLine : *this )
		if( string::npos == sLine.find(sSubString) ) tfResult.push_back( sLine );

	clear();
	*this = tfResult;
}

sizeint CTextFile::ExcludeWithout( const string& sSubString )
{
	CTextFile tfResult;
	for( auto& sLine : *this )
		if( string::npos != sLine.find(sSubString) ) tfResult.push_back( sLine );

	clear();
	*this = tfResult;

	sizeint siRowsLeft = this->size();
	return siRowsLeft;
}

void CTextFile::GetUnique( string_v& vsResult )
{
	map_si msiContent;
	for( auto& sLine : *this ) msiContent[ sLine ] = 1;

	vsResult.clear();
	for( map_si::const_iterator itMap = msiContent.begin(); itMap != msiContent.end(); ++itMap )
	{
		vsResult.push_back( itMap->first );
	}
}

void CTextFile::LeaveUnique()
{
	GetUnique( *this );
}

void CTextFile::ExcludeEachFromList( const string_v& vsList )
{
	CTextFile tfResult;
	for( auto& sLine : *this )
	{
		bool bExists = false;
		for( auto& sListLine : vsList )
		{
			if( sLine == sListLine ) {
				bExists = true;
				break;
			}
		}

		if( !bExists ) tfResult.push_back( sLine );
	}

	*this = tfResult;
}

void CTextFile::LeaveTextInQuotes( const string sQuote /*= "\""*/ )
{
	CTextFile tfResult;
	for( auto& sLine : *this )
	{
		do
		{
			size_t nPos = sLine.find( sQuote );
			if( string::npos == nPos ) break;
			sLine.erase( 0, ++nPos );

			nPos = sLine.rfind( sQuote );
			if( string::npos == nPos ) break;
			sLine.erase( nPos );
		}
		while( false );
		tfResult.push_back( sLine );
	}

	*this = tfResult;
}

void CTextFile::Sort()
{
	typedef multimap<string&,bool> MMStorage;

	MMStorage mmSortStorage;
	for( auto& sLine : *this )
	{
		mmSortStorage.insert( MMStorage::value_type(sLine, true) );
	}

	CTextFile tfResult;
	for( MMStorage::const_iterator itMap = mmSortStorage.begin(); itMap != mmSortStorage.end(); ++itMap )
	{
		tfResult.push_back( itMap->first );
	}

	*this = tfResult;
}
