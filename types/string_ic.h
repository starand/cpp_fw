#ifndef __STRING_IGNORE_CASE__
#define __STRING_IGNORE_CASE__

#include "strutils.h"
#include "macroes.h"

#ifdef LINUX
#	include <strings.h>
#	define stricmp strcasecmp
#endif

class string_ic:
	public string
{
public:
	string_ic() {};
	string_ic(const string &sValue) : string(sValue) {};
	string_ic(const char *pszValue) : string(pszValue) {};

	bool operator==(const string_ic &sValue) const 
	{ 
		return stricmp(this->c_str(), sValue.c_str()) == 0;
	}

	bool operator!=(const string_ic &sValue) const 
	{ 
		return !operator ==(sValue); 
	}

	bool operator<(const string_ic &sValue) const 
	{ 
		return stricmp(this->c_str(), sValue.c_str()) < 0; 
	}

	bool operator<=(const string_ic &sValue) const 
	{
		return !sValue.operator<(*this); 
	}

	bool operator>(const string_ic &sValue) const 
	{
		return sValue.operator<(*this);
	}

	bool operator>=(const string_ic &sValue) const 
	{
		return !operator<(sValue); 
	}

	size_t find( const string& str, size_t pos = 0 ) const
	{
		string sMain( c_str() ), sSearch( str.c_str() );
		char* szMain = (char*)sMain.c_str();
		char* szSearch = (char*)sSearch.c_str();

		for_each( sMain, nMainLen, i ) { szMain[i] = tolower( szMain[i] ); }
		for_each( sSearch, nSearchLen, i ) { szSearch[i] = tolower( szSearch[i] ); }
		return sMain.find( sSearch, pos );
	}
};

#endif // __STRING_IGNORE_CASE__
