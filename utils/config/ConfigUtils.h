#ifndef __CONFIGUTILS_H_INCLUDED
#define __CONFIGUTILS_H_INCLUDED

#include "ConfigOptions_Enum.h"
#include "ConfigStorage.h"


void AddConfigString(ECONFIGOPTION coConfigOption, const string &sOptionValue);

bool GetConfigString(ECONFIGOPTION coConfigOption, string &sOptionValue);
void GetConfigStringDef(ECONFIGOPTION coConfigOption, string &sOptionValue, const string &sDefValue);

template<typename INTTYPE>
bool GetConfigInt(ECONFIGOPTION coConfigOption, INTTYPE &nOptionValue)
{
	extern CConfigStorage g_csConfigStorage;
	return g_csConfigStorage.GetOptionInt(coConfigOption, nOptionValue);
}

template<typename INTTYPE>
void GetConfigIntDef(ECONFIGOPTION coConfigOption, INTTYPE &nOptionValue, INTTYPE nDefValue)
{
	extern CConfigStorage g_csConfigStorage;
	if (!g_csConfigStorage.GetOptionInt(coConfigOption, nOptionValue))
	{
		nOptionValue = nDefValue;
	}
}

#endif // __CONFIGUTILS_H_INCLUDED
