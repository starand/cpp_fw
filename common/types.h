#ifndef __TYPES__
#define __TYPES__

#include "platform.h"

#ifdef WINDOWS

#	include <windows.h>
#	define ALIGNED(n) __declspec(align(n))

typedef __int64 int64;
typedef unsigned __int64 uint64;

#elif defined(LINUX)

#	include <inttypes.h>

//typedef int64_t int64;
typedef long long int int64;
//typedef uint64_t uint64;
typedef unsigned long long int uint64;


#endif

#include <string>
#include <vector>
#include <map>


#define BITSINBYTE	8

using namespace std;

typedef unsigned short ushort;
typedef unsigned long ulong;
typedef unsigned int uint;
typedef unsigned char uchar;


typedef vector<string> string_v;
typedef vector<string_v> string_vv;

typedef vector<wstring> wstring_v;
typedef vector<wstring_v> wstring_vv;

typedef vector<uint> uint_v;
typedef vector<ulong> ulong_v;

typedef map<string, string> map_ss;
typedef map<string, int> map_si;

typedef uchar byte;
typedef size_t sizeint;

#endif // __TYPES__
