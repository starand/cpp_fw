#ifndef __CONFIGFILE_H_INCLUDED
#define __CONFIGFILE_H_INCLUDED

#include "types.h"
#include "string_ic.h"

typedef map<string_ic, string> CSectionData;
typedef map<string_ic, CSectionData* > CIniSettingStorage;

class CConfigFile
{
public:
	CConfigFile();
	CConfigFile( const char* szFileName );

	virtual ~CConfigFile();

public:
	bool LoadFromFile(const string& sFileName, bool bHideError = false);
	void Clear();

public:
	bool GetOptionValue(const string& sSection, const string& sParam, string& sValue);
	void GetOptionValue(const string& sSection, const string& sParam, string& sValue, const char* szDefaultValue);

	bool GetOptionValue(const string& sSection, const string& sParam, int& nValue);
	void GetOptionValue(const string& sSection, const string& sParam, int& nValue, int nDefValue);

	bool GetOptionValue(const string& sSection, const string& sParam, ushort& nValue);
	void GetOptionValue(const string& sSection, const string& sParam, ushort& nValue, ushort usDefValue);

	void SetOptionValue(const string& sSection, const string& sParam, const string& sValue);

public:
	void Debug();

private:
	void RemoveBrackets(string& sVarValue) const;
	void ParseKeyValue(const string& sInputValue, string& sOutKey, string& sOutValue) const;

private:
	CIniSettingStorage	m_issStorage;
};

#define config_t CConfigFile;

#endif // __CONFIGFILE_H_INCLUDED
