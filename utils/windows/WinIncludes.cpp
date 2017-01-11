#include "StdAfx.h"

#include "WinIncludes.h"

#include "Environment.h"
#include "WinRegister.h"

#define PERC_CHAR	'%'

static const char g_szCommon7ToolsFolder[] = "\\Common7\\Tools\\";
static const string g_sVcInclude = "VC\\include";
static const string g_sVSWow6432Path = "SOFTWARE\\Wow6432Node\\Microsoft\\VisualStudio\\";
static const string g_sShellFolder = "ShellFolder";
static const string g_sWinKitRegPath = "SOFTWARE\\Microsoft\\Windows Kits\\Installed Roots";
static const string g_sKitRoot = "KitsRoot";
static const string g_sKitIncludeUm = "Include\\um";
static const string g_sKitIncludeShared = "Include\\shared";

static const string g_asVersionsArray[EVSV__MAX] =
{
	"",

	"8.0",
	"9.0",
	"10.0",
	"11.0",
};

const string &EncodeVSVersion(sizeint siVersion)
{
	ASSERTE(IN_RANGE(siVersion, EVSV_VS80, EVSV__MAX));

	return g_asVersionsArray[siVersion];
}


static const string g_asVSEnvironmentVariables[EVSV__MAX] =
{
	"",

	"VS80COMNTOOLS",	// 2005
	"VS90COMNTOOLS",	// 2008
	"VS100COMNTOOLS",	// 2010
	"VS110COMNTOOLS",	// 2012
};

const string &EncodeVisualStudioVersionEnv(sizeint siVersion)
{
	ASSERTE(IN_RANGE(siVersion, EVSV_VS80, EVSV__MAX));

	return g_asVSEnvironmentVariables[siVersion];
}

static const string g_asStudioNames[EVSV__MAX] =
{
	"",

	"Visual Studio 2005",
	"Visual Studio 2008",
	"Visual Studio 2010",
	"Visual Studio 2012",
};


/*static */
EVISUALSTUDIOVERSION CWinIncludes::GetLatestInstalledVersionEnv()
{
	string sInstallationPath;
	return GetLatestInstallPathEnv(sInstallationPath);
}

/*static */
EVISUALSTUDIOVERSION CWinIncludes::GetLatestInstallPathEnv(string &sOutPath)
{
	EVISUALSTUDIOVERSION vsvResult = EVSV_NOTINSTALLED;

	for (sizeint siVersion = EVSV_VS110; siVersion > EVSV__MIN; --siVersion)
	{
		const string &sVSEnvVar = EncodeVisualStudioVersionEnv(siVersion);

		string sEnvVarValue;
		if (CEnvironment::GetVariable(sVSEnvVar, sEnvVarValue) && sEnvVarValue[0] != PERC_CHAR)
		{
			sizeint siCommon7ToolsPos = sEnvVarValue.find(g_szCommon7ToolsFolder);
			if (string::npos != siCommon7ToolsPos)
			{
				sEnvVarValue.erase(siCommon7ToolsPos + 1);
			}

			sOutPath = sEnvVarValue;
			vsvResult = (EVISUALSTUDIOVERSION)siVersion;
			break;
		}
	}

	return vsvResult;
}

/*static */
bool CWinIncludes::GetLatestIncludePathEnv(string &sOutPath)
{
	bool bResult = false;

	string sInstallationFolder;
	if (GetLatestInstallPathEnv(sInstallationFolder) != EVSV_NOTINSTALLED)
	{
		sOutPath = sInstallationFolder + g_sVcInclude;
		bResult = true;
	}

	return bResult;
}


/*static */
EVISUALSTUDIOVERSION CWinIncludes::GetLatestInstalledVersionReg()
{
	EVISUALSTUDIOVERSION svVersion = EVSV_NOTINSTALLED;

	for (sizeint siVersion = EVSV_VS110; siVersion != EVSV_NOTINSTALLED; --siVersion)
	{
		if (CheckIfVersionExistsReg((EVISUALSTUDIOVERSION)siVersion))
		{
			svVersion = (EVISUALSTUDIOVERSION)siVersion;
			break;
		}
	}

	return svVersion;
}

/*static */
bool CWinIncludes::CheckIfVersionExistsReg(EVISUALSTUDIOVERSION vsVersion)
{
	const string &siCurrentVersion = EncodeVSVersion(vsVersion);
	string sKeyPath = g_sVSWow6432Path + siCurrentVersion;

	bool bResult = CWinRegister::CheckPathExists(HKEY_LOCAL_MACHINE, sKeyPath);
	return bResult;
}


/*static */
bool CWinIncludes::GetLatestInstallPathReg(string &sOutPath)
{
	bool bResult = false;

	do
	{
		EVISUALSTUDIOVERSION vsLastVersion = GetLatestInstalledVersionReg();
		if (EVSV_NOTINSTALLED == vsLastVersion)
		{
			break;
		}

		const string &siCurrentVersion = EncodeVSVersion(vsLastVersion);
		string sKeyPath = g_sVSWow6432Path + siCurrentVersion;

		string sVSInstallPath;
		if (!CWinRegister::GetKeyValue(HKEY_LOCAL_MACHINE, sKeyPath, g_sShellFolder, sVSInstallPath))
		{
			break;
		}

		sOutPath = sVSInstallPath;
		bResult = true;
	}
	while (false);

	return bResult;
}

/*static */
bool CWinIncludes::GetLatestIncludePathReg(string &sOutPath)
{
	bool bResult = false;

	string sInstallPath;
	if (GetLatestInstallPathReg(sInstallPath))
	{
		sOutPath = sInstallPath + g_sVcInclude;
		bResult = true;
	}

	return bResult;
}


/*static */
bool CWinIncludes::GetWinKitInstallPath(string &sOutPath)
{
	bool bResult = false;

	string sWinKitPath;
	if (CWinRegister::GetKeyValue(HKEY_LOCAL_MACHINE, g_sWinKitRegPath, g_sKitRoot, sWinKitPath))
	{
		sOutPath = sWinKitPath;
		bResult = true;
	}

	return bResult;
}

/*static */
bool CWinIncludes::GetWinKitIncludePathsCollection(string_v &vsOutPathsCollection)
{
	bool bResult = false;
	string_v vsPathsCollection;

	string sWinKitPath;
	if (GetWinKitInstallPath(sWinKitPath))
	{
		string sIncludeUm = sWinKitPath + g_sKitIncludeUm;
		vsPathsCollection.push_back(sIncludeUm);

		string sIncludeShared = sWinKitPath + g_sKitIncludeShared;
		vsPathsCollection.push_back(sIncludeShared);

		vsPathsCollection.swap(vsOutPathsCollection);
		bResult = true;
	}

	return bResult;
}

#include <winapifamily.h>