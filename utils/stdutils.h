#ifndef __H_STDUTILS__
#define __H_STDUTILS__

#include "types.h"

#define BLACK		0
#define BLUE		1
#define GREEN		2
#define AQUA		3
#define RED			4
#define PURPLE		5
#define YELLOW		6
#define LIGHTGRAY	7
#define GRAY		8
#define LIGHTBLUE	9
#define LIGHTGREEN	10
#define LIGHTAQUA	11
#define LIGHTRED	12
#define LIGHTPURPLE	13
#define LIGHTYELLOW	14
#define WHITE		15

namespace StdUtils
{
	bool SetTextAttribs( ushort usColor ); 
	ushort GetTextAttribs();

	extern ushort usSavedAttribs;
	void SaveTextAttribs( ushort& usAttribs = usSavedAttribs );
	void RestoreTextAttribs( ushort& usAttribs = usSavedAttribs );

	void ShowError( const char* szMessage, bool bEndl = true );

	class outcolor
	{
	public:
		outcolor( const char* szText, ushort usColor );
	};

	ostream& save( ostream& stream );
	ostream& restore( ostream& stream );

	ostream& black( ostream& stream );
	ostream& blue( ostream& stream );
	ostream& green( ostream& stream );
	ostream& aqua( ostream& stream );
	ostream& red( ostream& stream );
	ostream& purple( ostream& stream );
	ostream& yellow( ostream& stream );
	ostream& lightgray( ostream& stream );
	ostream& gray( ostream& stream );
	ostream& lightblue( ostream& stream );
	ostream& lightgreen( ostream& stream );
	ostream& lightaqua( ostream& stream );
	ostream& lightred( ostream& stream );
	ostream& lightpurple( ostream& stream );
	ostream& lightyellow( ostream& stream );
	ostream& white( ostream& stream );

	class CColorSaver
	{
	public:
		CColorSaver() { SaveTextAttribs( m_usColor ); }
		~CColorSaver() { RestoreTextAttribs( m_usColor ); }
		ushort m_usColor;
	};

	class spaces
	{
	public:
		spaces( size_t nCount ) : m_nSpaces(nCount) { }
		friend ostream& operator<<( ostream& stream, spaces& ob );
	private:
		size_t m_nSpaces;
	};

#define AUTOCOLORSAVE CColorSaver csTMPSaver;

#ifdef SAVE_COLOR
	AUTOCOLORSAVE
#endif


#ifndef SHOW_ERROR
#	define SHOW_ERROR(x) cout << StdUtils::save << StdUtils::lightred << (x) << StdUtils::restore;
#endif

	void CreateCommandLineParamsVactor(int argc, char *argv[], string_v &vsOutParams);
	void CreateCommandLineParamsVactor(char *szCommandLine, string_v &vsOutParams);
};

#endif // __H_STDUTILS__
