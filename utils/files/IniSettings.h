#ifndef __H_INIPARSER__
#define __H_INIPARSER__

#include "types.h"
#include "string_ic.h"

typedef map<string_ic, string> CSectionData;
typedef map<string_ic, CSectionData* > CIniSettingStorage;

class CIniSettings
{
public:
	CIniSettings();
	CIniSettings( const char* szFileName );
	virtual ~CIniSettings();

	bool LoadFromFile(const string& sFileName, bool bHideError = false);
	void Clear();

	bool GetSectionValue(const string& sSection, const string& sParam, string& sValue);
	void GetSectionValue(const string& sSection, const string& sParam, string& sValue, const char* szDefaultValue);

	bool GetSectionValue(const string& sSection, const string& sParam, int& nValue);
	void GetSectionValue(const string& sSection, const string& sParam, int& nValue, int nDefValue);

	bool GetSectionValue(const string& sSection, const string& sParam, ushort& nValue);
	void GetSectionValue(const string& sSection, const string& sParam, ushort& nValue, ushort usDefValue);

	void SetSectionValue(const string& sSection, const string& sParam, const string& sValue);

	CSectionData& operator[]( const string& sSectionName );
	
	void Debug();

protected:
	virtual void SetDefaults() { }

private:
	CIniSettingStorage	m_issStorage;
};

#endif // __H_INIPARSER__
