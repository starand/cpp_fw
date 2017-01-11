#ifndef __H_FILEUTILS__
#define __H_FILEUTILS__

#include "types.h"
#include <ios>


#define DELFILE_OK			1
#define DELFILE_NOTFOUND	-1
#define DELFILE_ERROR		0

#define CHKSTR_IGNORECASE	0x00000001
#define CHKSTR_OPPOSEXPR	0x00000002
#define CHKSR_CHKALLINLINE	0x00000004
#define CHKSTR_DEFAULT	( CHKSR_CHKALLINLINE )

extern const char szDefaultFilter[];

namespace FileUtils
{

	enum EFILEACCESSTIME
	{
		EFAT_CREATE,
		EFAT_ACCESS,
		EFAT_WRITE
	};

	bool GetFileContent( const char* szFileName, string& sFileData );
	inline bool GetFileContent(const string& sFileName, string& sFileData)
	{
		return GetFileContent(sFileName.c_str(), sFileData);
	}

	bool PutFileContent( const char* szFileName, const char* szFileData, 
		size_t nFileLen,  int nOpenMode = ios::out );
	inline bool PutFileContent(const string& sFileName, const string& sFileData, int nOpenMode = ios::out | ios::binary)
	{
		return PutFileContent(sFileName.c_str(), sFileData.c_str(), sFileData.length(), nOpenMode);
	}

	bool GetFileData( const char* szFileName, char* szData, size_t& nDataLen, const char* szOpenMode = "rb" );
	bool PutFileData( const char* szFileName, const char* szData, size_t nDataLen, const char* szOpenMode = "wb" );

	bool GetBinaryFullName( string& sResult );
	bool GetBinaryDir( string& sResult );

#ifdef WINDOWS
	bool GetCurrentDir( string& sCurrentDir );
	const char* GetCurrentDir();


	size_t GetFolderList( const string& sPath, vector<string>& vsResultList, 
		const char* szFilter = szDefaultFilter, bool bRecursive = false );
	size_t GetFileList( const string& sPath, vector<string>& vsResultList, 
		const char* szFilter = szDefaultFilter, bool bClear = true, bool bFullPath = false );
	size_t GetFileListRecursive( const char* szFolder, string_v& vsResult, bool bClear = true, const char* szFilter = szDefaultFilter );

	bool GetListByPattern( const char* szPattern, string_v& vsList );
	bool GetDirectoryContent( const char* szFolder, string_v& vsResult, 
		const char* szFilter = szDefaultFilter, bool bCompletePath = false );
	bool GetDirectoryContent( const char* szFolder, string_v& vsFoldersList, string_v& vsFilesList, 
		const char* szFilter = szDefaultFilter, bool bCompletePath = false, bool bClear = true );
#endif

	bool GetFileStrings( const char* szFileName, string_v& vsStrings, 
		bool bIgnoreEmpty = true, bool bClear = true );
	inline bool GetFileStrings(const string& sFileName, string_v& vsStrings, bool bIgnoreEmpty = true, bool bClear = true)
	{
		return GetFileStrings(sFileName.c_str(), vsStrings, bIgnoreEmpty, bClear);
	}

	bool PutFileStrings( const char* szFileName, const string_v& vsStrings, bool bEndl = true);

	bool CheckStringsInFile( const string_v& vsStrings, const string_v& vsFileData, 
		uint_v& viResLines, ulong ulParams = CHKSTR_DEFAULT );
	bool CheckStringsInLine( const string& sLine, const string_v& vsStrings, 
		ulong ulParams = CHKSTR_DEFAULT );

	bool ReplaceInFile( const char* szFileName, const char* szFrom, const char* szTo );
	bool GetFileSize( const char* szFileName, uint64& nFileSize, const char* szOpenMode = "r" );
	inline bool GetFileSize(const string& sFileName, uint64& nFileSize, const char* szOpenMode = "r") { return GetFileSize(sFileName.c_str(), nFileSize, szOpenMode); }

#ifdef WINDOWS
	bool GetFileTime( const string& sFileName, string& sTime, EFILEACCESSTIME efatType = EFAT_CREATE );
	bool ConvertFiletimeToString( const FILETIME& ftTime, string& sResult );
	bool GetLastWriteTime( const string& sFileName, string& sResult );
	bool GetLastWriteTime( const string& sFileName, FILETIME& ftTime );
	bool GetCreateTime( const string& sFileName, string& sResult );
	bool GetCreatesTime( const string& sFileName, FILETIME& ftTime );
	bool GetLastAccessTime( const string& sFileName, string& sResult );
	bool GetLastAccessTime( const string& sFileName, FILETIME& ftTime );
	bool GetFileTimes( const string& sFileName, string& sCreateTime, string& sAccessTime, string& sWriteTime );

	bool SetLastWriteTime(const string& sFileName, const FILETIME& ftLastWriteTime);

	int Delete( const char* szFileName, bool bForceDelete = false );
	size_t DeleteList( const string_v& vsList, bool bForceDelete = false );

	bool DeleteDirectory( const char* szDirName, bool bForceDelete = false, bool bRecursive = false );
	bool CreateFolder( const char* szFolderName );

	bool CreateFolderPath(const char* szFolderName);
	inline bool CreateFolderPath(const string& sFolderPath) { return CreateFolderPath(sFolderPath.c_str()); }

	uint GetAttribs( const char* szName );
	bool SetAttribs( const char* szName, uint dwFileAttribs = FILE_ATTRIBUTE_NORMAL );

	bool IsDirectory( const char* szName );
	bool IsReadOnly( const char* szName );
	bool IsSystem( const char* szName );
	bool IsHidden( const char* szName );
	
	bool Exists( const char* szName );
	bool Exists( const string& sName );

	bool CleanFile( const char* szFileName );
#endif

	bool Rename( const string& sOldFileName, const string& sNewFileName );
	bool CopyFile( const string& sFromName, const string& sToName );


class CFileWrapper
{
public:
	CFileWrapper(const char *szFileName, const char *szFileMode);
	~CFileWrapper();

	bool IsOpen() const { return m_fpFile != NULL; }

	operator FILE*() const { return m_fpFile; }
	operator bool() const { return IsOpen(); }

private:
	FILE	*m_fpFile;
};

};

namespace PathUtils 
{
	bool SplitFilePath( const char* szFilePath, string& sFileName, string& sPath );
	const char* GetShortFileName( const char* szFilePath );
	string GetFolderPart( const string& sPath );

	size_t ParseCommandLine( const char* szCommandLine, string_v& vsResult );

	const char* DelExt( char* szFileName );
	string& DelExt( string& sFileName );
	string& ReplaceExt( string& sFileName, const char* szNexExt );
	const char* GetExt( const char* szFileName, bool bWithDot = false );

	string& NormaliseSlashes( string& sPath, bool bWindowsFormat = true );
	string& DeleteDoubleSlashes( string& sPath );

#ifdef WINDOWS
	string& CanonicalizePath(string& sPath);
	string& NormalizePath( string& sPath, bool bWindowsFormat = true );

	string GetPathDifference( string& sFirstFolder, string& sSecondFolder );
	size_t GetFoldersCount( const string& sPath, bool bFile = false );
#endif
}

#endif // __H_FILEUTILS__
