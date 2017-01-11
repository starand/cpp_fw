#ifndef __H_LOGERROR__
#define __H_LOGERROR__

#ifdef WINDOWS
#	define snprintf sprintf_s
#else
#	include <stdio.h>
#endif


#ifdef __H_STDUTILS__
#	define _LIGHTAQUA_	StdUtils::lightaqua
#	define _LIGHTRED_	StdUtils::lightred
#	define _LIGHTGRAY_	StdUtils::lightgray
#else
#	define _LIGHTAQUA_	""
#	define _LIGHTRED_	""
#	define _LIGHTGRAY_	""
#endif

#define ERROR_BUFFER_SIZE 512

#define FORMAT_MESSAGE1(x, p1) \
	char szErrorBuf[ERROR_BUFFER_SIZE] = { 0 }; \
	snprintf( szErrorBuf, ERROR_BUFFER_SIZE - 1, x, p1 );

#define FORMAT_MESSAGE2(x, p1, p2) \
	char szErrorBuf[ERROR_BUFFER_SIZE] = { 0 }; \
	snprintf( szErrorBuf, ERROR_BUFFER_SIZE - 1, x, p1,p2 );

#define FORMAT_MESSAGE3(x, p1, p2, p3) \
	char szErrorBuf[ERROR_BUFFER_SIZE] = { 0 }; \
	snprintf( szErrorBuf, ERROR_BUFFER_SIZE - 1, x, p1,p2,p3 );


#define S(x) #x
#define S_(x) S(x)
#define S__LINE__ S_(__LINE__)


#ifndef LOG_PREFIX
#	define LOG_PREFIX ""
#endif

#ifndef LOG_SUFFIX
#	define LOG_SUFFIX ""
#endif

#ifndef CONSOLE_PREFIX
#	define CONSOLE_PREFIX ""
#endif

#ifndef CONSOLE_SUFFIX
#	define CONSOLE_SUFFIX ""
#endif


/////////////////////////////////////////////////////////
// CONSOLE LOGGING

#ifdef _CONSOLE
#	define LOG_MSG_CONSOLE(x) cout << _LIGHTAQUA_ << CONSOLE_PREFIX << x << CONSOLE_SUFFIX << _LIGHTGRAY_ << endl;
#	define LOG_TRACE_CONSOLE(x) cout << _LIGHTGRAY_ << CONSOLE_PREFIX << x << CONSOLE_SUFFIX << _LIGHTGRAY_ << endl;
#	define LOG_ERROR_CONSOLE(x) cout << _LIGHTAQUA_ << CONSOLE_PREFIX << _LIGHTRED_ << x << _LIGHTAQUA_ << CONSOLE_SUFFIX << _LIGHTGRAY_ << endl;
#else
#	define LOG_MSG_CONSOLE(x)
#	define LOG_TRACE_CONSOLE(x)
#	define LOG_ERROR_CONSOLE(x)
#endif


/////////////////////////////////////////////////////////
// MESSAGE LOGGING

#ifdef __LOGFILE_H_INCLUDED
#	define LOG_MSG(x) if( g_pLog ) g_pLog->AddMessage( (x), __FUNCTION__, __FILE__, __LINE__ ); LOG_MSG_CONSOLE(x);
#elif !defined( LOG_MSG )
#	define LOG_MSG(x) LOG_MSG_CONSOLE(x)
#endif

#define LOG_MSG2(x,p1) { FORMAT_MESSAGE1(x,p1); LOG_MSG( szErrorBuf ); }
#define LOG_MSG3(x,p1,p2) { FORMAT_MESSAGE2(x, p1, p2); LOG_MSG( szErrorBuf ); }
#define LOG_MSG4(x,p1,p2,p3) { FORMAT_MESSAGE3(x, p1, p2, p3); LOG_MSG( szErrorBuf ); }


/////////////////////////////////////////////////////////
// ERROR LOGGING

#ifdef __LOGFILE_H_INCLUDED
#	define LOG_ERROR(x) if( g_pLog ) g_pLog->AddMessage( (x), __FUNCTION__, __FILE__, __LINE__ ); LOG_ERROR_CONSOLE(x);
#elif !defined( LOG_ERROR )
#	define LOG_ERROR(x) LOG_ERROR_CONSOLE(x)
#endif

#define LOG_ERROR_EXIT(x) LOG_ERROR((x)); exit( -1 );
#define LOG_ERROR2(x,p1) { FORMAT_MESSAGE1(x,p1); LOG_ERROR( szErrorBuf ); }
#define LOG_ERROR3(x,p1,p2) { FORMAT_MESSAGE2(x, p1, p2); LOG_ERROR( szErrorBuf ); }
#define LOG_ERROR4(x,p1,p2,p3) { FORMAT_MESSAGE3(x, p1, p2, p3); LOG_ERROR( szErrorBuf ); }

#define LOG_ERROR_BREAK(x) { LOG_ERROR(x); break; }
#define LOG_ERROR2_BREAK(x,p1) { LOG_ERROR2(x,p1); break; }
#define LOG_ERROR3_BREAK(x,p1,p2) { LOG_ERROR3(x,p1,p2); break; }
#define LOG_ERROR4_BREAK(x,p1,p2,p3) { LOG_ERROR4(x,p1,p2,p3); break; }


/////////////////////////////////////////////////////////
// TRACE LOGGING

#ifdef __LOGFILE_H_INCLUDED
#	define LOG_TRACE(x) if( g_pLog ) g_pLog->AddMessage( (x), __FUNCTION__, __FILE__, __LINE__ ); LOG_TRACE_CONSOLE(x);
#elif !defined( LOG_TRACE )
#	define LOG_TRACE(x) LOG_TRACE_CONSOLE(x)
#endif

#define LOG_TRACE2(x,p1) { FORMAT_MESSAGE1(x,p1); LOG_TRACE( szErrorBuf ); }
#define LOG_TRACE3(x,p1,p2) { FORMAT_MESSAGE2(x, p1, p2); LOG_TRACE( szErrorBuf ); }
#define LOG_TRACE4(x,p1,p2,p3) { FORMAT_MESSAGE3(x, p1, p2, p3); LOG_TRACE( szErrorBuf ); }



#define _ASSERT_BREAK( x ) _ASSERT( (x) ); if( !(x) ) break;
#define _ASSERT_MEMORY_BREAK( x ) if( !(x) ) { _ASSERT( (x) ); LOG_ERROR_BREAK( szUnableToAllocateMemory ); }

#endif // __H_LOGERROR__
