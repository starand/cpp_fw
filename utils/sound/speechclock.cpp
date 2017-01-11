#include "stdafx.h"
#include "speechclock.h"
#include "datetime.h"
#include "fileutils.h"

//////////////////////////////////////////////////
// CSpeechClock implementation

void CSpeechClock::SayTime()
{
	static const char szTimeDelim[] = ":";
	static const char szSpeechFormat[] = "CreateObject(\"SAPI.SpVoice\").Speak\"%s\"";
	static const char szFileName[] = "./timespeech.vbs";
	
	do
	{
		string_v vsTimeParts;
		split( CDateTime::Now(EDTF_TIME), vsTimeParts, szTimeDelim );
		if( vsTimeParts.size() != 3 ) break;

		string& sHour =  vsTimeParts[ 0 ];
		string& sMinute =  vsTimeParts[ 1 ];
		if( sHour[0] == '0' ) sHour.erase( 0, 1 );
		if( sMinute[0] == '0' ) 
		{
			sMinute.erase( 0, 1 );
		}
		else
		{
			if( sMinute[0] != '1' && sMinute[1] != '0' )
			{
				sMinute.insert( 1, "0 " );
			}
		}

		string sSpeechString, sSpeechFileContent;
		FormatString( sSpeechString, "%s hour %s minutes.", vsTimeParts[0].c_str(), vsTimeParts[1].c_str() );
		FormatString( sSpeechFileContent, szSpeechFormat, sSpeechString.c_str() );
		
		if( !FileUtils::PutFileContent(szFileName, sSpeechFileContent.c_str(), sSpeechFileContent.length()) )
		{
			break;
		}

		system( "wscript timespeech.vbs" );
	}
	while( false );
}