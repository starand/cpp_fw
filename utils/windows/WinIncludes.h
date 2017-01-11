#ifndef __VISUALSTUDIOLOCATION_H_INCLUDED
#define __VISUALSTUDIOLOCATION_H_INCLUDED


enum EVISUALSTUDIOVERSION
{
	EVSV__MIN,
	EVSV_NOTINSTALLED = EVSV__MIN,

	EVSV_VS80,		// VS2005
	EVSV_VS90,		// vs2008
	EVSV_VS100,		// VS2010
	EVSV_VS110,		// VS2012

	EVSV__MAX,
};

class CWinIncludes
{
public:
	static EVISUALSTUDIOVERSION GetLatestInstalledVersionEnv();
	static EVISUALSTUDIOVERSION GetLatestInstallPathEnv(string &sOutPath);
	static bool GetLatestIncludePathEnv(string &sOutPath);

public:
	static EVISUALSTUDIOVERSION GetLatestInstalledVersionReg();
	static bool CheckIfVersionExistsReg(EVISUALSTUDIOVERSION vsVersion);

	static bool GetLatestInstallPathReg(string &sOutPath);
	static bool GetLatestIncludePathReg(string &sOutPath);

public:
	static bool GetWinKitInstallPath(string &sOutPath);
	static bool GetWinKitIncludePathsCollection(string_v &vsOutPathsCollection);
};

#endif // __VISUALSTUDIOLOCATION_H_INCLUDED
