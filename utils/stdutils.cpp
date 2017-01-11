#include "StdAfx.h"
#include "stdutils.h"

#ifdef SAVEINPUTCOLOR
	CColorSaver csTMPSaver;
#endif

#define SPACE	' '
#define TAB		'\t'
#define QUOTE	'\"'

namespace StdUtils {

bool SetTextAttribs( ushort usColor )
{
	static HANDLE hStdOut = GetStdHandle( STD_OUTPUT_HANDLE   );
	return ( SetConsoleTextAttribute( hStdOut, usColor ) != 0 );
}

ushort GetTextAttribs()
{
	CONSOLE_SCREEN_BUFFER_INFO csbi;
	static HANDLE hStdOut = GetStdHandle( STD_OUTPUT_HANDLE   );
	GetConsoleScreenBufferInfo( hStdOut, &csbi );
	return csbi.wAttributes;
}

ushort usSavedAttribs;
void SaveTextAttribs( ushort& usAttribs )
{
	usAttribs = GetTextAttribs();
}

void RestoreTextAttribs( ushort& usAttribs )
{
	SetTextAttribs( usAttribs ); 
}

void ShowError( const char* szMessage, bool bEndl )
{
	if( !szMessage || *szMessage == 0 ) return;

	SaveTextAttribs();
	cout << lightred << szMessage;
	if( bEndl ) cout << endl;
	RestoreTextAttribs();
}

/////////////////////////////////////////////////////////
// outcolor implementation

outcolor::outcolor( const char* szText, ushort usColor )
{
	SaveTextAttribs();
	SetTextAttribs( usColor );
	cout << szText;
	RestoreTextAttribs();
}

//////////////////////////////////////////
// color manipulators implementation

ushort usSavedColor = GetTextAttribs();
ostream& save( ostream& stream ) { usSavedColor = GetTextAttribs(); return stream; }
ostream& restore( ostream& stream ) { SetTextAttribs( usSavedColor ); stream.flush(); return stream; }

ostream& black( ostream& stream ) { SetTextAttribs( BLACK ); stream.flush(); return stream; }
ostream& blue( ostream& stream ) { SetTextAttribs( BLUE ); stream.flush(); return stream; }
ostream& green( ostream& stream ) { SetTextAttribs( GREEN ); stream.flush(); return stream; }
ostream& aqua( ostream& stream ) { SetTextAttribs( AQUA ); stream.flush(); return stream; }
ostream& red( ostream& stream ) { SetTextAttribs( RED ); stream.flush(); return stream; }
ostream& purple( ostream& stream ) { SetTextAttribs( PURPLE ); stream.flush(); return stream; }
ostream& yellow( ostream& stream ) { SetTextAttribs( YELLOW ); stream.flush(); return stream; }
ostream& lightgray( ostream& stream ) { SetTextAttribs( LIGHTGRAY ); stream.flush(); return stream; }
ostream& gray( ostream& stream ) { SetTextAttribs( GRAY ); stream.flush(); return stream; }
ostream& lightblue( ostream& stream ) { SetTextAttribs( LIGHTBLUE ); stream.flush(); return stream; }
ostream& lightgreen( ostream& stream ) { SetTextAttribs( LIGHTGREEN ); stream.flush(); return stream; }
ostream& lightaqua( ostream& stream ) { SetTextAttribs( LIGHTAQUA ); stream.flush(); return stream; }
ostream& lightred( ostream& stream ) { SetTextAttribs( LIGHTRED ); stream.flush(); return stream; }
ostream& lightpurple( ostream& stream ) { SetTextAttribs( LIGHTPURPLE ); stream.flush(); return stream; }
ostream& lightyellow( ostream& stream ) { SetTextAttribs( LIGHTYELLOW ); stream.flush(); return stream; }
ostream& white( ostream& stream ) { SetTextAttribs( WHITE ); stream.flush(); return stream; }


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


ostream& operator<<( ostream& stream, spaces& ob )
{
	for( size_t idx = 0; idx < ob.m_nSpaces; ++idx ) stream << ' ';
	return stream;
}


void CreateCommandLineParamsVactor(int argc, char *argv[], string_v &vsOutParams)
{
	ASSERTE(argv && argc);
	string_v vsParams;

	for (int idx = 0; idx < argc; ++idx)
	{
		vsParams.push_back(argv[idx]);
	}

	vsOutParams.swap(vsParams);
}

void CreateCommandLineParamsVactor(char *szCommandLine, string_v &vsOutParams)
{
	ASSERTE(szCommandLine);
	string_v vsParams;

	bool bInQuotes = false;

	sizeint siCommandLineLen = strlen(szCommandLine);
	char *pszStartPos = const_cast<char *>(szCommandLine);

	for (sizeint siIndex = 0; siIndex < siCommandLineLen; ++siIndex)
	{
		if (szCommandLine[siIndex] == QUOTE)
		{
			bInQuotes = !bInQuotes;
		}

		if (!bInQuotes && (szCommandLine[siIndex] == SPACE || szCommandLine[siIndex] == TAB))
		{
			if (pszStartPos == szCommandLine + siIndex)
			{
				++pszStartPos;
			}
			else
			{
				szCommandLine[siIndex] = 0;

				vsParams.push_back(pszStartPos);
				pszStartPos = szCommandLine + siIndex + 1;
			}
		}
	}

	if (strlen(pszStartPos))
	{
		vsParams.push_back(pszStartPos);
	}

	vsOutParams.swap(vsParams);
}

}; // namespace StdUtils
