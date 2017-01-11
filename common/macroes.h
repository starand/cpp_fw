#ifndef __H_MACROES__
#define __H_MACROES__

#define INVALID_RESULT		-1

#define FREE_POINTER(x) { delete (x); (x) = NULL; }
#define FREE_POINTER_ASSERT(x) { ASSERT(x != NULL); delete (x); (x) = NULL; }
#define FREE_POINTER_ARRAY(x) if( (x) ) { delete [] (x); (x) = NULL; }

#define CLOSE_FILE(x) if( (x) ) { if(x) fclose(x); (x) = NULL; }

// iterate through STL container
#define for_each( x,_size_, ind  ) \
		size_t _size_ = x.size(); \
		for( size_t ind = 0; ind < _size_; ++ind )

#define VAR_DUMP(x) \
	for_each( (x), nLen, i ) cout << (x)[i].c_str() << endl;

#ifdef DEBUG_INFO_SHOWTIMER
#	define START_TIMER(x) int64 x = GetTickCount64();
#	define SHOW_TIME(x) cout << GetTickCount64() - x << endl;
#else
#	define START_TIMER(x)
#	define SHOW_TIME(x)
#endif

#define ARRAY_SIZE(A) ( sizeof(A) / sizeof((A)[0]) )

#define START_FUNCTION() do {
#define END_FUNCTION() } while( false );
#define END_FUNCTION_RET(x) } while( false ); return (x);

#define BOOL_FUNCTION_START() \
	bool bResult = false; do {
#define START_FUNCTION_BOOL() BOOL_FUNCTION_START()

#define BOOL_FUNCTION_END() \
	bResult = true; } while( false ); \
	return bResult;
#define END_FUNCTION_BOOL() BOOL_FUNCTION_END()

#define BOOL_FUNCTION_END_ACTION(x) \
	bResult = true; } while( false ); \
	x; return bResult;

#define _MIN(a,b) (((a) > (b)) ? (b) : (a))
#define _MAX(a,b) (((a) < (b)) ? (b) : (a))

#define IN_RANGE(Value, Min, Max) ((size_t)((size_t)(Value) - (size_t)(Min)) < (size_t)((size_t)(Max) - (size_t)(Min)))

#ifdef _CONSOLE
#	define CONSOLE_WRITE(_X_) cout << _X_;
#	define CONSOLE_WRITELN(_X_) cout << _X_ << endl;
#else
#	define CONSOLE_WRITE(_X_)
#	define CONSOLE_WRITELN(_X_)
#endif

#define CHECK_ANY_FAULT() if(bAnyFault) break;
#define CHECK_RETURN(_VALUE_) if (!(_VALUE_)) return ;

#define atol64(_VAL_) static_cast<uint64>(_atoi64(_VAL_))


#define ALIGN_UP(Size, Alignment)	(((size_t)(Size) + ((Alignment) - 1))&  ~(size_t)((Alignment) - 1))

#endif // __H_MACROES__
