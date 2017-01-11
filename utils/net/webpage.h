#ifndef __H_WEBPAGE__
#define __H_WEBPAGE__

#define PAGE_SIZE_LIMIT 1024 * 1024		// 1 Mbyte

extern const char szEmptyURL[];

class CWebPage
{
public:
	CWebPage( const char* szPageUrl = szEmptyURL );
	~CWebPage();

	bool LoadPage( const char* szPageUrl, size_t nMaxPageSize = PAGE_SIZE_LIMIT );
	size_t ParseUrlsByFileExt( const char* szFileExt, string_v& vsUrls );

	bool GetIsPageLoaded() const;

	bool SaveToFile( const char* szFileName ) const;
	bool LoadFromFile( const char* szFileName );

	const string& GetPageContent() const;

private:
	string	m_sUrl;
	string	m_sPageContent;
	bool	m_bPageLoaded;
};

class CUrlList :
	public vector<string>
{
public:
	size_t DowloadAndSaveToFolder( const string& sDestFolder );
private:
};

#endif // __H_WEBPAGE__
