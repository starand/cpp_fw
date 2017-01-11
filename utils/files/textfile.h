#ifndef __H_TEXTFILE__
#define __H_TEXTFILE__

//////////////////////////////////////////////////
// CTextFile definition

class CTextFile : public string_v
{
public:
	CTextFile() { }
	CTextFile( const string& sFileName );

	bool LoadFromFile( const string& sFileName, bool bClear = true );
	bool SaveToFile( const string& sFileName ) const;

	void GetLinesWith( const string& sSubString, string_v& vsResult, bool bClean = true );
	void GetLinesWith( const string_v& vsSubStrList, string_v& vsResult, bool bClean = true );
	void GetLinesWithout( const string& sSubString, string_v& vsResult, bool bClean = true );
	void GetUnique( string_v& vsResult );

	void ExcludeWith( const string& sSubString );
	sizeint ExcludeWithout( const string& sSubString );
	void ExcludeEachFromList( const string_v& vsList );

	void LeaveUnique();
	void LeaveTextInQuotes( const string sQuote = "\"" );

	void Sort();

};

#endif // __H_TEXTFILE__
