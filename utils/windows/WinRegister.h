#ifndef __WINREGISTER_H_INCLUDED
#define __WINREGISTER_H_INCLUDED

//////////////////////////////////////////////////
// CWinRegister declaration

class CWinRegister
{
public:
	CWinRegister();
	~CWinRegister();

public:
	static bool GetKeyValue(HKEY hRoot, const string &sKeyPath, const string &sKeyName, string &sOutKeyValue);

	static bool CheckPathExists(HKEY hRoot, const string &sKeyPath);

private:

};

#endif // __WINREGISTER_H_INCLUDED
