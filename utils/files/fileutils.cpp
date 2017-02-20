#include "StdAfx.h"
#include "fileutils.h"
#include <fstream>
#include "macroes.h"
#include "string_ic.h"
#include "istring.h"
#include "strutils.h"
#include "consts.h"

#include <cstring>


#ifdef WINDOWS
#	include <io.h>
#	include <windows.h>
#	include <Shlwapi.h>
#	include <direct.h>
#	pragma comment(lib, "Shlwapi.lib")
#else
#	include <unistd.h>
#endif


#ifdef OUTPUT_PROGRESS
#	include "stdutils.h"
#endif


#ifdef WINDOWS
const char szFolderSeparator[] = "\\";
#else
const char szFolderSeparator[] = "/";
#endif

const char szDefaultFilter[] = "*.*";


namespace FileUtils {

bool GetFileContent( const char* szFileName, string& sFileData )
{
	bool bResult = false;
	sFileData.clear();

	do
	{
		ifstream fin( szFileName, ios::ate);
		if( fin.fail() ) break;

		int nFileLen = (int)fin.tellg();
		fin.close();

		fin.open(szFileName, ios::in);
		if (!fin.good()) break;

		sFileData.resize( nFileLen );
		fin.read( const_cast<char*>(sFileData.c_str()), nFileLen );
		fin.close();

		bResult = true;
	}
	while(false);

	return bResult;
}

bool PutFileContent( const char* szFileName, const char* szFileData,
	size_t nFileLen, int nOpenMode /* = ios::out */  )
{
	bool bResult = false;
	ofstream fout;

	do
	{
		if( !szFileName || !szFileName[0] ) break;

		fout.open( szFileName, (ios_base::openmode)nOpenMode );
		if( fout.fail() ) break;

		fout.write( szFileData, nFileLen );
		if( fout.fail() ) break;
		bResult = true;
	}
	while (false);

	if( fout.is_open() )
	{
		fout.close();
	}

	return bResult;
}

#ifdef WINDOWS
bool GetBinaryFullName( string& sResult )
{
	bool bResult = false;

	do
	{
		sResult.clear();
		sResult.resize( MAX_PATH, 0 );
		HMODULE hModule = GetModuleHandle(NULL);
		if( !hModule ) break;

		if( !GetModuleFileNameA(hModule, (char *)sResult.c_str(), MAX_PATH) ) break;

		bResult = true;
	}
	while(false);

	return bResult;
}
#else
bool GetBinaryFullName(string& binary)
{
	const size_t buffer_size = 1024;
	char buffer[buffer_size];

	int res = readlink("/proc/self/exe", buffer, buffer_size);
	if (res == -1)
	{
		return false;
	}
	buffer[res] = 0;

	binary = buffer;
	return true;
}
#endif

bool GetBinaryDir( string& sResult )
{
#ifdef Q_OS_ANDROID
    sResult = "./";
    return true;
#endif
	bool bResult = false;

	do
	{
		if( !GetBinaryFullName( sResult ) ) break;

		size_t nPos = sResult.rfind( szFolderSeparator );
		if( string::npos == nPos ) break;

		sResult.erase( ++nPos );

		bResult = true;
	}
	while(false);

	return bResult;
}

#ifdef WINDOWS
bool GetCurrentDir( string& sCurrentDir )
{
	bool bResult = false;
	do
	{
		static char szBuffer[ MAX_PATH ] = { 0 };
		if( !GetCurrentDirectoryA(MAX_PATH, szBuffer) ) break;
		sCurrentDir.assign(szBuffer);
		bResult = true;
	}
	while(false);
	return bResult;
}

const char* GetCurrentDir()
{
	static char szBuffer[ MAX_PATH ] = { 0 };
	if( !GetCurrentDirectoryA(MAX_PATH, szBuffer) ) return NULL;
	return szBuffer;
}


size_t GetFolderList( const string& sPath, vector<string>& vsResultList,
	const char* szFilter, bool bRecursive )
{
	struct _finddatai64_t fdData;
	string sFolder = sPath + szFolderSeparator + szFilter;

	if( !bRecursive ) vsResultList.clear();

	intptr_t ipHandle = _findfirsti64( sFolder.c_str(), &fdData );
	if( ipHandle >= 0 )
	{
		do
		{
			if( !(fdData.attrib & _A_SUBDIR)
				|| strcmp(fdData.name,".") == 0 || strcmp(fdData.name,"..") == 0 ) continue;

			vsResultList.push_back( sPath + szFolderSeparator + fdData.name );

			if( bRecursive )
			{
				GetFolderList( sPath + szFolderSeparator + fdData.name, vsResultList, szFilter, bRecursive );
			}
		}
		while( _findnexti64( ipHandle, &fdData ) == 0 );
		_findclose( ipHandle );
	}

	return vsResultList.size();
}

size_t GetFileList( const string& sPath, vector<string>& vsResultList, const char* szFilter, bool bClear, bool bFullPath /*= false*/ )
{
	struct _finddatai64_t fdData;
	string sFolder = sPath + szFolderSeparator + szFilter;
	if( bClear ) vsResultList.clear();

	intptr_t ipHandle = _findfirsti64( sFolder.c_str(), &fdData );
	if( ipHandle >= 0 )
	{
		do
		{
			if( fdData.attrib & _A_SUBDIR ) continue;

			if( bFullPath ) vsResultList.push_back( sPath + szFolderSeparator + fdData.name );
			else vsResultList.push_back( fdData.name );
		}
		while( _findnexti64( ipHandle, &fdData ) == 0 );
		_findclose( ipHandle );
	}

	return vsResultList.size();
}

size_t GetFileListRecursive( const char* szFolder, string_v& vsResult, bool bClear /*= true*/, const char* szFilter /*= szDefaultFilter*/ )
{
	if( bClear ) vsResult.clear();

	string_v vsFolders;
	vsFolders.push_back( szFolder );
	GetFolderList(szFolder, vsFolders, szDefaultFilter, true);

	for_each( vsFolders, nFolderCount, idx )
		GetFileList( vsFolders[idx].c_str(), vsResult, szFilter, false, true );

	return vsResult.size();
}


bool GetListByPattern( const char* szPattern, string_v& vsList )
{
	string sFolder( ".\\" );
	string sPattern( szPattern );

	for_each( sPattern, nLen, i ) if( sPattern[i] == cSlash ) sPattern[i] = cBackSlash;

	size_t nSlashPos = sPattern.find_last_of( szSlashes );
	if( string::npos != nSlashPos )
	{
		sFolder = sPattern.substr( 0, ++nSlashPos );
		sPattern.erase( 0, nSlashPos );
	}

	return GetDirectoryContent( sFolder.c_str(), vsList, sPattern.c_str(), true );
}

bool GetDirectoryContent( const char* szFolder, string_v& vsResult, const char* szFilter, bool bCompletePath )
{
	bool bResult = false;
	do
	{
		struct _finddatai64_t fdData;
		string sFolder = string( szFolder ) + cSlash;

		char cLastChar = sFolder[ sFolder.length() - 1 ];
		if(  cLastChar != cSlash && cLastChar != cBackSlash ) sFolder += szFolderSeparator;

		vsResult.clear();
		intptr_t ipHandle = _findfirsti64( (sFolder + szFilter).c_str(), &fdData );
		if( ipHandle == -1 ) break;

		do
		{
			if( fdData.attrib & _A_SUBDIR )
			{
				if( !strcmp(fdData.name, ".") || !strcmp(fdData.name, "..") ) continue;

				vsResult.push_back( (bCompletePath ? sFolder : string()) + fdData.name + "/" );
			}
			else
				vsResult.push_back( (bCompletePath ? sFolder : string()) + fdData.name );
		}
		while( _findnexti64( ipHandle, &fdData ) == 0 );
		_findclose( ipHandle );

		bResult = true;
	}
	while( false );
	return bResult;
}

bool GetDirectoryContent( const char* szFolder, string_v& vsFoldersList, string_v& vsFilesList, const char* szFilter, bool bCompletePath, bool bClear )
{
	bool bResult = false;
	do
	{
		struct _finddatai64_t fdData;
		string sFolder = string( szFolder ) + cSlash;

		if( bClear )
		{

			vsFoldersList.clear();
			vsFilesList.clear();
		}

		intptr_t ipHandle = _findfirsti64( (sFolder + szFilter).c_str(), &fdData );
		if( ipHandle == -1 ) break;

		do
		{
			if( fdData.attrib & _A_SUBDIR )
			{
				if( !strcmp(fdData.name, ".") || !strcmp(fdData.name, "..") ) continue;
				vsFoldersList.push_back( (bCompletePath ? sFolder : string()) + fdData.name + "/" );
			}
			else
			{
				vsFilesList.push_back( (bCompletePath ? sFolder : string()) + fdData.name );
			}
		}
		while( _findnexti64( ipHandle, &fdData ) == 0 );
		_findclose( ipHandle );

		bResult = true;
	}
	while( false );
	return bResult;
}
#endif

bool GetFileStrings( const char* szFileName, string_v& vsStrings, bool bIgnoreEmpty, bool bClear )
{
	bool bResult = false;
	if( bClear ) vsStrings.clear();

	do
	{
		string sFileLine;
		ifstream fin( szFileName );

		if( !fin.is_open() ) break;

		while( !fin.eof() && fin.good() )
		{
			getline( fin, sFileLine );
			if( bIgnoreEmpty && sFileLine.empty() ) continue;

			vsStrings.push_back( sFileLine );
		}

		fin.close();
		bResult = true;
	}
	while (false);

	return bResult;
}

bool PutFileStrings( const char* szFileName, const string_v& vsStrings, bool bEndl )
{
	bool bResult = false;

	do
	{
		ofstream fout( szFileName );
		if( !fout.is_open() ) break;

		for_each( vsStrings, nStringsCount, idx )
		{
			fout << vsStrings[ idx ].c_str();
			if( bEndl ) fout << endl;
		}

		fout.close();
		bResult = true;
	}
	while (false);

	return bResult;
}

bool CheckStringsInFile( const string_v& vsStrings, const string_v& vsFileData, uint_v& viResLines, ulong ulParams )
{
	bool bResult = true;

	bool bIgnoreCase = (ulParams & CHKSTR_IGNORECASE) > 0;
	bool bOpposite = (ulParams & CHKSTR_OPPOSEXPR) > 0;

	size_t nFileDataSize = vsFileData.size();
	size_t nStringsSize = vsStrings.size();

	for( size_t nStringsPos = 0; nStringsPos < nStringsSize; ++nStringsPos )
	{
		size_t nPos = 0;
		bool bCurLineExists = false;
		for( size_t nFilePos = 0; nFilePos < nFileDataSize; ++nFilePos )
		{
			if( bIgnoreCase ) nPos = ((istring*)&vsFileData[nFilePos])->find(vsStrings[nStringsPos].c_str());
			else nPos = vsFileData[nFilePos].find(vsStrings[nStringsPos]);

			bool bExists = ( !bOpposite ? string::npos != nPos : string::npos == nPos );
			if( bExists )
			{
				viResLines.push_back( nFilePos );
				bCurLineExists = true;
			}
		}

		if( !bCurLineExists )
		{
			bResult = false;
			break;
		}
	}

	return bResult;
}

bool CheckStringsInLine( const string& sLine, const string_v& vsStrings, ulong ulParams )
{
	bool bAllInLine = (ulParams & CHKSR_CHKALLINLINE) > 0;
	bool bIgnoreCase = (ulParams & CHKSTR_IGNORECASE) > 0;
	bool bOpposite = (ulParams & CHKSTR_OPPOSEXPR) > 0;

	bool bResult = bAllInLine;
	do
	{
		size_t nStringsCount = vsStrings.size();
		if( !nStringsCount ) break;

		size_t nPos =  0;
		for( size_t idx = 0; idx < nStringsCount; ++idx )
		{
			if( bIgnoreCase ) nPos = ((istring*)&sLine)->find( vsStrings[idx].c_str() );
			else nPos = sLine.find( vsStrings[idx] );
			bool bNotExists = ( !bOpposite ? string::npos == nPos  : string::npos != nPos );
			if( bNotExists )
			{
				if( bAllInLine )
				{
					bResult = false;
					break;
				}
			}
			else
			{
				if( !bAllInLine )
				{
					bResult = true;
					break;
				}
			}
		}
	}
	while( false );
	return bResult;
}


bool ReplaceInFile( const char* szFileName, const char* szFrom, const char* szTo )
{
	bool bResult = false;
	do
	{
		string sFileData;
		if( !GetFileContent(szFileName, sFileData) ) break;

		size_t nPos = 0;
		while( string::npos != (nPos = sFileData.find(szFrom, nPos)) )
		{
			sFileData.replace( nPos, strlen(szFrom), szTo );
		}

		if( !PutFileContent(szFileName, sFileData.c_str(), sFileData.length()) ) break;
		bResult = true;
	}
	while(false);
	return bResult;
}

bool GetFileSize( const char* szFileName, uint64& nFileSize, const char* szOpenMode )
{
	bool bResult = false;

	FILE* fp = NULL;
	do
	{
#ifdef WINDOWS
		if( fopen_s(&fp, szFileName, szOpenMode) ) break;
#else
		fp = fopen(szFileName, szOpenMode);
#endif
		if( !fp || fseek(fp, 0, SEEK_END) ) break;

		uint64 nCurSize = ftell( fp );
		if( nCurSize == (uint64)-1L ) break;

		nFileSize = nCurSize;
		bResult = true;
	}
	while(false);

	if( fp ) fclose(fp);
	return bResult;
}

#ifdef WINDOWS

bool GetFileTime( const string& sFileName, string& sTime, EFILEACCESSTIME efatType /*= EFAT_CREATE */)
{
	START_FUNCTION_BOOL();

	if( sFileName.empty() ) break;

	WIN32_FILE_ATTRIBUTE_DATA lpFileInformation;
	if( 0 == GetFileAttributesExA(sFileName.c_str(), GetFileExInfoStandard, &lpFileInformation) ) break;

	FILETIME filetime;
	switch( efatType )
	{
	case EFAT_CREATE:
		filetime = lpFileInformation.ftCreationTime;
		break;
	case EFAT_ACCESS:
		filetime = lpFileInformation.ftLastAccessTime;
		break;
	case EFAT_WRITE:
		filetime = lpFileInformation.ftLastWriteTime;
		break;
	}

	if( !ConvertFiletimeToString(filetime, sTime) ) break;

	END_FUNCTION_BOOL();
}

bool GetFileTimes( const string& sFileName, string& sCreateTime, string& sAccessTime, string& sWriteTime )
{
	START_FUNCTION_BOOL();

	if( sFileName.empty() ) break;

	WIN32_FILE_ATTRIBUTE_DATA lpFileInformation;
	if( 0 == GetFileAttributesExA(sFileName.c_str(), GetFileExInfoStandard, &lpFileInformation) ) break;

	FILETIME ftCreationTime = lpFileInformation.ftCreationTime;
	FILETIME ftAccessTime = lpFileInformation.ftLastAccessTime;
	FILETIME ftWriteTime = lpFileInformation.ftLastWriteTime;

	if( !ConvertFiletimeToString(ftCreationTime, sCreateTime) ) break;
	if( !ConvertFiletimeToString(ftAccessTime, sAccessTime) ) break;
	if( !ConvertFiletimeToString(ftWriteTime, sWriteTime) ) break;

	END_FUNCTION_BOOL();
}


bool SetLastWriteTime(const string& sFileName, const FILETIME& ftLastWriteTime)
{
	BOOL_FUNCTION_START();

	HANDLE hFile = CreateFileA(sFileName.c_str(), GENERIC_READ | FILE_WRITE_ATTRIBUTES, FILE_SHARE_READ, NULL, OPEN_EXISTING, 0, NULL);
	if (INVALID_HANDLE_VALUE == hFile)
	{
		break;
	}

	bool bAnyFault = !SetFileTime(hFile, NULL, NULL, &ftLastWriteTime);

	CloseHandle(hFile);

	CHECK_ANY_FAULT();
	BOOL_FUNCTION_END();
}


bool ConvertFiletimeToString( const FILETIME& ftTime, string& sResult )
{
	START_FUNCTION_BOOL();

	sResult.erase();

	SYSTEMTIME stUTC, stLocal;
	if( 0 == FileTimeToSystemTime(&ftTime, &stUTC) ) break;
	if( 0 == SystemTimeToTzSpecificLocalTime(NULL, &stUTC, &stLocal) ) break;

	const sizeint siBufferSize = 64;
	sResult.reserve(siBufferSize);
	sprintf_s(const_cast<char *>(sResult.data()), siBufferSize,
		szTimeForamt, stLocal.wMonth, stLocal.wDay, stLocal.wYear, stLocal.wHour, stLocal.wMinute, stLocal.wSecond );

	END_FUNCTION_BOOL();
}

bool GetLastWriteTime( const string& sFileName, string& sResult )
{
	return GetFileTime( sFileName.c_str(), sResult, EFAT_WRITE );
}

bool GetLastCreateTime( const string& sFileName, FILETIME& ftTime )
{
	START_FUNCTION_BOOL();

	if( sFileName.empty() ) break;
	WIN32_FILE_ATTRIBUTE_DATA lpFileInformation;
	if( 0 == GetFileAttributesExA(sFileName.c_str(), GetFileExInfoStandard, &lpFileInformation) ) break;
	ftTime = lpFileInformation.ftCreationTime;

	END_FUNCTION_BOOL();
}

bool GetLastAccessTime( const string& sFileName, FILETIME& ftTime )
{
	START_FUNCTION_BOOL();

	if( sFileName.empty() ) break;
	WIN32_FILE_ATTRIBUTE_DATA lpFileInformation;
	if( 0 == GetFileAttributesExA(sFileName.c_str(), GetFileExInfoStandard, &lpFileInformation) ) break;
	ftTime = lpFileInformation.ftLastAccessTime;

	END_FUNCTION_BOOL();
}

bool GetLastWriteTime( const string& sFileName, FILETIME& ftTime )
{
	START_FUNCTION_BOOL();

	if( sFileName.empty() ) break;
	WIN32_FILE_ATTRIBUTE_DATA lpFileInformation;
	if( 0 == GetFileAttributesExA(sFileName.c_str(), GetFileExInfoStandard, &lpFileInformation) ) break;
	ftTime = lpFileInformation.ftLastWriteTime;

	END_FUNCTION_BOOL();
}

bool GetCreateTime( const string& sFileName, string& sResult )
{
	return GetFileTime( sFileName.c_str(), sResult, EFAT_CREATE );
}

bool GetLastAccessTime( const string& sFileName, string& sResult )
{
	return GetFileTime( sFileName.c_str(), sResult, EFAT_ACCESS );
}


int Delete( const char* szItemName, bool bForceDelete  )
{
	uint dwAttribs = ::GetFileAttributesA( szItemName );
	if( INVALID_FILE_ATTRIBUTES == dwAttribs ) return DELFILE_NOTFOUND;

	if( bForceDelete )
	{
		dwAttribs &= !( FILE_ATTRIBUTE_ARCHIVE | FILE_ATTRIBUTE_READONLY | FILE_ATTRIBUTE_SYSTEM );
		::SetFileAttributesA( szItemName, dwAttribs );
	}

	int nRes = 0;
	if( FILE_ATTRIBUTE_DIRECTORY & dwAttribs ) nRes = _rmdir( szItemName );
	else nRes = remove( szItemName );
	return ( nRes == 0 ? DELFILE_OK : DELFILE_ERROR );
}

size_t DeleteList( const string_v& vsList, bool bForceDelete /*= false*/ )
{
	size_t nResult = 0;
	for_each( vsList, nListSize, idx )
	{
		if( DELFILE_OK == Delete(vsList[idx].c_str(), bForceDelete) == 1 ) ++nResult;
	}

	return nResult;
}

bool IsDirectory( const char* szName )
{
	uint dwAttribs = ::GetFileAttributesA( szName );
	if( INVALID_FILE_ATTRIBUTES == dwAttribs ) return false;
	else return ( dwAttribs & FILE_ATTRIBUTE_DIRECTORY ) > 0;
}

bool IsReadOnly( const char* szName )
{
	uint dwAttribs = ::GetFileAttributesA( szName );
	if( INVALID_FILE_ATTRIBUTES == dwAttribs ) return false;
	else return ( dwAttribs & FILE_ATTRIBUTE_READONLY ) > 0;
}

bool IsSystem( const char* szName )
{
	uint dwAttribs = ::GetFileAttributesA( szName );
	if( INVALID_FILE_ATTRIBUTES == dwAttribs ) return false;
	else return ( dwAttribs & FILE_ATTRIBUTE_SYSTEM ) > 0;
}

bool IsHidden( const char* szName )
{
	uint dwAttribs = ::GetFileAttributesA( szName );
	if( INVALID_FILE_ATTRIBUTES == dwAttribs ) return false;
	else return ( dwAttribs & FILE_ATTRIBUTE_HIDDEN ) > 0;
}

uint GetAttribs( const char* szName )
{
	return ::GetFileAttributesA( szName );
}

bool SetAttribs( const char* szName, uint dwFileAttribs )
{
	return ( 0 != SetFileAttributesA(szName, dwFileAttribs) );
}

bool Exists( const char* szName )
{
	uint dwAttribs = ::GetFileAttributesA( szName );
	return ( INVALID_FILE_ATTRIBUTES != dwAttribs );
}

bool Exists( const string& sName )
{
	uint dwAttribs = ::GetFileAttributesA( sName.c_str() );
	return ( INVALID_FILE_ATTRIBUTES != dwAttribs );
}

bool DeleteDirectory( const char* szDirName, bool bForceDelete, bool bRecursive )
{
	bool bResult = true;

	if( bRecursive )
	{
		string_v sDirsList, sFilesList;
		GetDirectoryContent( szDirName, sDirsList, sFilesList );

		string sDir( szDirName );
		for_each( sDirsList, nDirsCount, idx )
		{
			if( DELFILE_OK != DeleteDirectory((sDir + szSlash + sDirsList[idx]).c_str(), bForceDelete, bRecursive) ) bResult = false;
		}

		for_each( sFilesList, nFilesCount, idx )
		{
			if( DELFILE_OK != Delete((sDir + szSlash + sFilesList[idx]).c_str(), bForceDelete) ) bResult = false;
		}

	}

	if( DELFILE_OK != Delete(szDirName) ) bResult = false;

	return bResult;
}

bool CreateFolder( const char* szFolderName )
{
	return ( CreateDirectoryA(szFolderName, NULL) > 0 );
}

bool CreateFolderPath( const char* szFolderName )
{
	bool bResult = true;
	string sPath( szFolderName );
	sPath += cBackSlash;
	PathUtils::NormaliseSlashes( sPath );

	size_t nStartPos = 0;
	if( sPath[1] == ':' ) nStartPos = 2;

	while( string::npos != (nStartPos = sPath.find(cBackSlash, ++nStartPos)) )
	{
		string sCurrentFolder = sPath.substr( 0, nStartPos );
		if( !Exists(sCurrentFolder.c_str()) && !CreateFolder(sCurrentFolder.c_str()) )
		{
			bResult = false;
			break;
		}
	}

	return bResult;
}

bool CleanFile( const char* szFileName )
{
	HANDLE hFile = CreateFileA( szFileName, GENERIC_WRITE, FILE_SHARE_WRITE, NULL, TRUNCATE_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL );
	bool bResult = ( INVALID_HANDLE_VALUE != hFile );
	if( bResult ) CloseHandle( hFile );
	return bResult;
}

#endif // WINDOWS


bool GetFileData( const char* szFileName, char* szData, size_t& nDataLen, const char* szOpenMode /*= "rb"*/ )
{
	bool bResult = false;

	FILE* fp = NULL;
	do
	{
#ifdef WINDOWS
		if (fopen_s(&fp, szFileName, szOpenMode)) break;
#else
		fp = fopen(szFileName, szOpenMode);
#endif
		if (!fp) break;

		nDataLen = fread( szData, 1, nDataLen, fp );
		if( ferror(fp) )  break;
		bResult = true;
	}
	while( false );
	if( fp ) fclose( fp );

	return bResult;
}

bool PutFileData( const char* szFileName, const char* szData, size_t nDataLen, const char* szOpenMode /*= "wb"*/ )
{
	bool bResult = false;

	FILE* fp = NULL;
	do
	{
#ifdef WINDOWS
		if (fopen_s(&fp, szFileName, szOpenMode)) break;
#else
		fp = fopen(szFileName, szOpenMode);
#endif
		if (!fp) break;
		if( nDataLen != fwrite(szData, 1, nDataLen, fp) ) break;
		bResult = true;
	}
	while( false );
	if( fp ) fclose( fp );

	return bResult;
}

bool Rename( const string& sOldFileName, const string& sNewFileName )
{
	return ( 0 == rename(sOldFileName.c_str(), sNewFileName.c_str()) );
}

bool CopyFile( const string& sFromName, const string& sToName )
{
	BOOL_FUNCTION_START();

	FILE* fpFrom = NULL, *fpTo = NULL;

#ifdef WINDOWS
	if( fopen_s(&fpFrom, sFromName.c_str(), "rb") || !fpFrom ) break;
	if( fopen_s(&fpTo, sToName.c_str(), "wb") || !fpTo ) break;
#else
	fpFrom = fopen(sFromName.c_str(), "rb"); if (!fpFrom) break;
	fpTo = fopen(sToName.c_str(), "wb"); if (!fpTo) break;
#endif

	char szBuffer[ 512 ] = { 0 };

	bool bErrorOcccured = false;
	do
	{
		bErrorOcccured = true;

		int nBytesRead = fread( szBuffer, 1, 512, fpFrom );
		if( ferror(fpFrom) ) break;

		if( nBytesRead ) {
			fwrite( szBuffer, 1, nBytesRead, fpTo );
			if( ferror(fpTo) ) break;
		}

		bErrorOcccured = false;
	}
	while( !feof(fpFrom) );

	if( bErrorOcccured ) break;
	BOOL_FUNCTION_END();
}


///////////////////////////////////////////////////
// CFileWrapper implementation

CFileWrapper::CFileWrapper(const char *szFileName, const char *szFileMode) : m_fpFile(NULL)
{
	START_FUNCTION();
	ASSERTE(szFileName && szFileMode && szFileMode[0] && szFileName[0]);

#ifdef WINDOWS
#pragma warning( disable : 4996 )
	m_fpFile = fopen(szFileName, szFileMode);
#pragma warning( default : 4996 )
#else
	m_fpFile = fopen(szFileName, szFileMode);
#endif

	if (!m_fpFile)
	{
		break;
	}

	END_FUNCTION();
}

CFileWrapper::~CFileWrapper()
{
	if (m_fpFile)
	{
		fclose(m_fpFile);
		m_fpFile = NULL;
	}
}

}; // namespace FileUtils


namespace PathUtils {

const char szFolderSeparators[] = "\\/";
const char szEmprtString[] = "";

const char cSlash = '/';
const char cBackSlash = '\\';
const char cDot = '.';
const char szSlashes[] = "\\/";

bool SplitFilePath( const char* szFilePath, string& sFileName, string& sPath )
{
	bool bResult = false;

	do
	{
		sPath.clear();
		sFileName.clear();
		string sFilePath( szFilePath );

		size_t nPos = sFilePath.find_last_of(szFolderSeparators);
		if( string::npos == nPos ) break;

		sFileName = sFilePath.substr( nPos + 1 );
		if( sFileName.empty() ) break;

		sPath = sFilePath.substr( 0, nPos );

		bResult = true;
	}
	while(false);

	return bResult;
}

const char* GetShortFileName( const char* szFilePath )
{
	static string sFilePath;

	sFilePath = szFilePath;
	size_t nLastSlahsPos = sFilePath.find_last_of(szFolderSeparators);
	if( string::npos == nLastSlahsPos ) nLastSlahsPos = 0;
	else ++nLastSlahsPos;

	return ( sFilePath.c_str() + nLastSlahsPos );
}

string GetFolderPart( const string& sPath )
{
	size_t nPos = sPath.find_last_of( szSlashes );
	if( string::npos == nPos ) return string();
	else return sPath.substr( 0, nPos );
}

string& DelBothQuotes( string& sValue)
{
	size_t nLen = sValue.length();
	if( nLen >= 2 && (sValue[0] == '\"' || sValue[0] == '\'') && sValue[nLen-1])
	{
		sValue.erase( nLen-1 );
		sValue.erase( 0, 1 );
	}

	return sValue;
}

size_t ParseCommandLine( const char* szCommandLine, string_v& vsResult )
{
	if( !szCommandLine || szCommandLine[0] == 0 ) return 0;
	vsResult.clear();

	bool bInQuotes = false, bSpaceBefore = false;

	size_t nCmdLineLen = strlen( szCommandLine ), nBeginPos = 0;
	for( size_t idx = 0; idx < nCmdLineLen; ++idx )
	{
		if( szCommandLine[idx] == '\"' )
		{
			bInQuotes = !bInQuotes;
			continue;
		}

		if( bInQuotes ) continue;

		if( szCommandLine[idx] == ' ' )
		{
			if( bSpaceBefore ) continue;
			else
			{
				string sParam = string(szCommandLine, nBeginPos, idx - nBeginPos);
				vsResult.push_back(DelBothQuotes(sParam));
				bSpaceBefore = true;
			}
		}
		else
		{
			if( bSpaceBefore )
			{
				nBeginPos = idx;
				bSpaceBefore = false;
			}
		}
	}

	if (!bSpaceBefore)
	{
		string sParam = string(szCommandLine, nBeginPos, string::npos);
		vsResult.push_back(DelBothQuotes(sParam));
	}

	return vsResult.size();
}

const char* DelExt( char* szFileName )
{
	size_t nLen = strlen( szFileName );
	if( nLen >= 5 && szFileName[ nLen - 4 ] == cDot ) szFileName[ nLen - 4 ] = 0;
	return szFileName;
}

string& DelExt( string& sFileName )
{
	size_t nPos = sFileName.rfind( cDot );
	size_t nLastSlashpos = sFileName.find_last_of( szSlashes );

	do
	{
		if( string::npos == nPos ) break;
		if( string::npos != nLastSlashpos && nLastSlashpos > nPos ) break;
		sFileName.erase( nPos );
	}
	while( false );

	return sFileName;
}

string& ReplaceExt( string& sFileName, const char* szNexExt )
{
	DelExt( sFileName );
	if( szNexExt ) sFileName.append( szNexExt );
	return sFileName;
}

const char* GetExt( const char* szFileName, bool bWithDot /*= false*/ )
{
	char* szResult = (char*)szEmprtString;
	if( szFileName )
	{
		size_t nLen = strlen( szFileName );
		for( int idx = nLen - 1; idx >= 0; --idx )
		{
			if( szFileName[idx] == cSlash || szFileName[idx] == cBackSlash ) break;
			if( szFileName[idx] == cDot )
			{
				szResult = (char*)szFileName + ( bWithDot ? idx : idx + 1 );
				break;
			}
		}
	}

	return szResult;
}


string& NormaliseSlashes( string& sPath, bool bWindowsFormat /*= true*/ )
{
	for_each( sPath, nPathLen, idx )
	{
		if( sPath[idx] != cSlash && sPath[idx] != cBackSlash ) continue;

		if( bWindowsFormat && sPath[idx] == cSlash ) sPath[idx] = cBackSlash;
		if( !bWindowsFormat && sPath[idx] == cBackSlash ) sPath[idx] = cSlash;

		if( idx > 0 && sPath[idx] == sPath[idx - 1] )
		{
			sPath.erase( idx, 1 );
			--nPathLen;
			--idx;
		}
	}

	return sPath;
}

string& DeleteDoubleSlashes( string& sPath )
{
	static const char szDoubleSlah[] = "//";
	static const char szDoubleBackSlah[] = "\\\\";

	size_t nPos = 0;
	while( string::npos != (nPos = sPath.find(szDoubleSlah)) ) sPath.erase( nPos, 1 );
	nPos = 0;
	while( string::npos != (nPos = sPath.find(szDoubleBackSlah)) ) sPath.erase( nPos, 1 );

	return sPath;
}

#ifdef WINDOWS
string& CanonicalizePath(string& sPath)
{
	char szBuffer[MAX_PATH + 1] = { 0 };

    //PathCanonicalizeA(szBuffer, sPath.c_str());
	sPath.assign(szBuffer);

	return sPath;
}

string& NormalizePath( string& sPath, bool bWindowsFormat /*= true*/ )
{
	DeleteDoubleSlashes( sPath );
	NormaliseSlashes(sPath, bWindowsFormat);
	CanonicalizePath(sPath);

	return sPath;
}

string GetPathDifference( string& sFirstFolder, string& sSecondFolder )
{
	string sResult;
	START_FUNCTION();

	PathUtils::NormalizePath( sFirstFolder );
	PathUtils::NormalizePath( sSecondFolder );

	size_t nFirstLen = sFirstFolder.length();
	size_t nSecondLen = sSecondFolder.length();
	if( nFirstLen == nSecondLen ) break;

	// second string will be longer
	string& sStringOne = ( nFirstLen > nSecondLen ? sSecondFolder : sFirstFolder );
	string& sStringTwo = ( nFirstLen < nSecondLen ? sSecondFolder : sFirstFolder );

	size_t nShorterLength = _MIN( nFirstLen, nSecondLen );
	string sSub = sStringTwo.substr(0, nShorterLength);
	if( sSub != sStringOne ) break;

	sResult = sStringTwo.substr( nShorterLength );
	END_FUNCTION_RET( sResult );
}

size_t GetFoldersCount( const string& sPath, bool bFile /*= false*/ )
{
	size_t nResult = 0;
	START_FUNCTION();

	if( sPath.empty() ) break;
	string sPathCopy = ( bFile ? GetFolderPart(sPath) : sPath );
	PathUtils::NormalizePath( sPathCopy );

	size_t nPos = sPathCopy.find( chColon );
	if( string::npos != nPos ) sPathCopy.erase( 0, nPos + 1 );
	if( sPathCopy.empty() ) break;

	if( sPathCopy[0] == cBackSlash ) sPathCopy.erase( 0, 1 );
	if( sPathCopy.empty() ) break;

	size_t nLen = sPathCopy.length();
	if( sPathCopy[nLen-1] == cBackSlash ) sPathCopy.erase( --nLen );
	if( sPathCopy.empty() ) break;

	++nResult;
	size_t nPrevPos = 0;
	while( string::npos != (nPos = sPathCopy.find(cBackSlash, nPrevPos)) )
	{
		nPrevPos = ++nPos;
		++nResult;
	}

	END_FUNCTION_RET( nResult );
}
#endif

}  // namespace PathUtils
