#include "StdAfx.h"
#include "WinRegister.h"
#include "windows.h"


#define BUFFER_SIZE		255


CWinRegister::CWinRegister()
{

}

CWinRegister::~CWinRegister()
{

}


/*static */
bool CWinRegister::GetKeyValue(HKEY hRoot, const string &sKeyPath, const string &sKeyName, string &sOutKeyValue)
{
	bool bResult = false;

	do 
	{
		HKEY hKey;
		if (ERROR_SUCCESS != RegOpenKeyExA(hRoot, sKeyPath.c_str(), NULL, KEY_ALL_ACCESS, &hKey))
		{
			break;
		}

		char szBuffer[BUFFER_SIZE];
		DWORD dwBufferSize = BUFFER_SIZE;
		DWORD dwType = REG_SZ;

		if (ERROR_SUCCESS != RegQueryValueExA(hKey, sKeyName.c_str(), NULL, &dwType, (LPBYTE)&szBuffer, &dwBufferSize))
		{
			break;
		}

		sOutKeyValue.assign(szBuffer);
		bResult = true;
	}
	while (false);
	
	return bResult;
}

/*static */
bool CWinRegister::CheckPathExists(HKEY hRoot, const string &sKeyPath)
{
	HKEY hKey;

	bool bResult = (ERROR_SUCCESS == RegOpenKeyExA(hRoot, sKeyPath.c_str(), NULL, KEY_READ, &hKey));
	return bResult;
}
