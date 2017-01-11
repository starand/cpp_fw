#include "StdAfx.h"
#include "webpage.h"
#include "httputils.h"
#include "fileutils.h"
#include "logerror.h"


static const char szEmptyURL[] = "";
static const char szQuote[] = "\"";

extern const char szErrorInvalidURL[];
static const char szErrorUnableToLoadPage[] = "Unable to load %s";
static const char szErrorUnableToSave[] = "Unable to save page into file %s";
static const char szErrorUnableToLoad[] = "Unable to load page from file %s";

static const char szInvalidExtension[] = "Invalid externsion";
static const char szInvalidFileName[] = "Invalid file name";
static const char szPageNotLoaded[] = "Page not loaded yet";
static const char szInvalidPageStruct[] = "Invalid page structure";

#define CHECK_IF_PAGE_LOADED() \
	if( !m_bPageLoaded ) { LOG_ERROR_BREAK( szPageNotLoaded ); }

//////////////////////////////////////////////////
// CWebPage implementation

CWebPage::CWebPage( const char* szPageUrl /*= szEmptyURL*/ ) 
	: m_sUrl( szPageUrl ? szPageUrl : szEmptyURL ), m_bPageLoaded()
{
	if( !m_sUrl.empty() ) 
	{
		m_bPageLoaded = LoadPage( m_sUrl.c_str() );
	}
}

CWebPage::~CWebPage()
{

}

bool CWebPage::LoadPage( const char* szPageUrl, size_t nMaxPageSize /*= PAGE_SIZE_LIMIT*/ )
{
	bool bResult = false;

	do
	{
		if( !szPageUrl || !szPageUrl[0] )
		{
			LOG_ERROR2_BREAK( szErrorInvalidURL, "empty or NULL" );
		}

		m_sUrl.assign( szPageUrl );
		if( !HttpUtils::DownloadPage( m_sUrl.c_str(), m_sPageContent, nMaxPageSize) )
		{
			LOG_ERROR2_BREAK( szErrorUnableToLoadPage, m_sUrl.c_str() );
		}

		bResult = true;
	}
	while( false );

	m_bPageLoaded = bResult;
	return bResult;
}

bool CWebPage::GetIsPageLoaded() const
{
	return m_bPageLoaded;
}

size_t CWebPage::ParseUrlsByFileExt( const char* szFileExt, string_v& vsUrls )
{
	size_t nResult = 0;

	do
	{
		CHECK_IF_PAGE_LOADED();
		if( !szFileExt || !szFileExt[0] )
		{
			LOG_ERROR_BREAK( szInvalidExtension );
		}

		size_t nPos = 0, nStartPos = 0;
		string sEndDelim = string( szFileExt ) + szQuote;
		while( string::npos != (nPos = m_sPageContent.find(sEndDelim, nPos)) )
		{
			nPos += strlen( szFileExt );
			// try too search begin quote
			nStartPos = m_sPageContent.rfind( szQuote, nPos - 1 );
			if( string::npos == nStartPos )
			{
				LOG_ERROR( szInvalidPageStruct );
				continue;
			}

			++nStartPos;
			string sUrl = m_sPageContent.substr( nStartPos, nPos - nStartPos );
			if( !sUrl.empty() )
			{
				vsUrls.push_back( sUrl );
				++nResult;
			}
		}
	}
	while( false );

	return nResult;
}

bool CWebPage::SaveToFile( const char* szFileName ) const
{
	bool bResult = true;

	do
	{
		CHECK_IF_PAGE_LOADED();
		if( !szFileName || !szFileName[0] )
		{
			LOG_ERROR_BREAK( szInvalidFileName );
		}

		if( !FileUtils::PutFileContent(szFileName, m_sPageContent.c_str(), m_sPageContent.size()) )
		{
			LOG_ERROR2_BREAK( szErrorUnableToSave, szFileName );
		}

		bResult = true;
	}
	while( false );

	return bResult;
}

bool CWebPage::LoadFromFile( const char* szFileName )
{
	bool bResult = false;

	do
	{
		if( !szFileName || !szFileName[0] )
		{
			LOG_ERROR_BREAK( szInvalidFileName );
		}

		if( !FileUtils::GetFileContent(szFileName, m_sPageContent) )
		{
			m_bPageLoaded = false;
			LOG_ERROR2_BREAK( szErrorUnableToLoad, szFileName );
		}

		bResult = m_bPageLoaded = true;
	}
	while( false );

	return bResult;
}

const string& CWebPage::GetPageContent() const
{
	return m_sPageContent;
}

//////////////////////////////////////////////////
// CUrlList implementation
size_t CUrlList::DowloadAndSaveToFolder( const string& sDestFolder )
{
	size_t nPagesLoaded = 0;
	if( FileUtils::CreateFolderPath(sDestFolder.c_str()) )
	{
		for( auto& sUrl : *this )
		{
			string sDestFile = sDestFolder + PathUtils::GetShortFileName( sUrl.c_str() );
			if( HttpUtils::DownloadFile(sUrl.c_str(), sDestFile.c_str()) )
			{
				++nPagesLoaded;
			}
		}
	}

	return nPagesLoaded;
}
