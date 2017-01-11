#ifndef __H_IGNORECASESTRING__
#define __H_IGNORECASESTRING__

#include <asserts.h>

#ifdef LINUX
#	include <wchar.h>
#else
#endif


template<class C>
struct char_traits_nocase : public std::char_traits<C>
{
	static bool eq( const C& c1, const C& c2 )
	{ 
		return ::toupper(c1) == ::toupper(c2); 
	}

	static bool lt( const C& c1, const C& c2 )
	{ 
		return ::toupper(c1) < ::toupper(c2);
	}

#ifdef WINDOWS
	static int compare( const C* s1, const C* s2, size_t N )
	{
		return _strnicmp(s1, s2, N);
	}
#endif

	static const char* find( const C* s, size_t N, const C& a )
	{
		for( size_t i=0 ; i<N ; ++i )
		{
			if( ::toupper(s[i]) == ::toupper(a) ) 
				return s+i ;
		}
		return 0 ;
	}

	static bool eq_int_type( const C& c1, const C& c2 )
	{ 
		return ::toupper(c1) == ::toupper(c2) ; 
	}		
};

template<>
struct char_traits_nocase<wchar_t> : public std::char_traits<wchar_t>
{
	static bool eq( const wchar_t& c1, const wchar_t& c2 )
	{ 
		return ::towupper(c1) == ::towupper(c2); 
	}

	static bool lt( const wchar_t& c1, const wchar_t& c2 )
	{ 
		return ::towupper(c1) < ::towupper(c2);
	}

	static int compare( const wchar_t* s1, const wchar_t* s2, size_t N )
	{
#ifdef WINDOWS
		return _wcsnicmp(s1, s2, N);
#else
		ASSERT_NOT_IMPLEMENTED();
		return wcscmp(s1, s2);
#endif
	}

	static const wchar_t* find( const wchar_t* s, size_t N, const wchar_t& a )
	{
		for( size_t i=0 ; i<N ; ++i )
		{
			if( ::towupper(s[i]) == ::towupper(a) ) 
				return s+i ;
		}
		return 0 ;
	}

	static bool eq_int_type( const int_type& c1, const int_type& c2 )
	{ 
		return ::towupper(c1) == ::towupper(c2) ; 
	}		
};

typedef std::basic_string<char, char_traits_nocase<char> > istring;
typedef std::basic_string<wchar_t, char_traits_nocase<wchar_t> > iwstring;
	
#endif // __H_IGNORECASESTRING__
