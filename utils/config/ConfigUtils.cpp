#include "StdAfx.h"

#include "ConfigUtils.h"
#include "ConfigStorage.h"

#include "fileutils.h"


/*extern */
CConfigStorage g_csConfigStorage;


void AddConfigString(ECONFIGOPTION coConfigOption, const string &sOptionValue)
{
	g_csConfigStorage.StoreOptionValue(coConfigOption, sOptionValue);
}


bool GetConfigString(ECONFIGOPTION coConfigOption, string &sOptionValue)
{
	return g_csConfigStorage.GetOptionString(coConfigOption, sOptionValue);
}

void GetConfigStringDef(ECONFIGOPTION coConfigOption, string &sOptionValue, const string &sDefValue)
{
	if (!g_csConfigStorage.GetOptionString(coConfigOption, sOptionValue))
	{
		sOptionValue = sDefValue;
	}
}
