#include "StdAfx.h"

#include "ConfigStorage.h"
#include "ConfigOptions_Enum.h"

//////////////////////////////////////////////////
// CConfigStorage implementation

CConfigStorage::CConfigStorage():
	m_osOptionsStorage()
{

}

CConfigStorage::~CConfigStorage()
{

}


void CConfigStorage::StoreOptionValue(ECONFIGOPTION coConfigOption, const string& sOptionValue)
{
	ASSERTE(IN_RANGE(coConfigOption, ECO__MIN, ECO__MAX));

	COptionsStorage &osOptionsStorage = GetOptionsStorage();
	auto insPair = osOptionsStorage.insert(make_pair(coConfigOption, sOptionValue));
	
	if (!insPair.second)
	{
		osOptionsStorage[coConfigOption] = sOptionValue;
	}
}


bool CConfigStorage::GetOptionString(ECONFIGOPTION coConfigOption, string &sOutOptionValue) const
{
	ASSERTE(IN_RANGE(coConfigOption, ECO__MIN, ECO__MAX));

	COptionsStorage &osOptionsStorage = GetOptionsStorage();
	COptionsStorage::const_iterator itOptionsMap = osOptionsStorage.find(coConfigOption);

	bool bResult = false;
	if (itOptionsMap != osOptionsStorage.end())
	{
		sOutOptionValue = itOptionsMap->second;
		bResult = true;
	}

	return bResult;
}


bool CConfigStorage::GetOptionInt(ECONFIGOPTION coConfigOption, int &nOutOptionValue) const
{
	string sStringValue;
	
	bool bOptionExists = GetOptionString(coConfigOption, sStringValue);

	if (bOptionExists)
	{
		nOutOptionValue = atoi(sStringValue.c_str());
	}

	bool bResult = bOptionExists;
	return bResult;
}

bool CConfigStorage::GetOptionInt(ECONFIGOPTION coConfigOption, sizeint &siOutOptionValue) const
{
	string sStringValue;

	bool bOptionExists = GetOptionString(coConfigOption, sStringValue);

	if (bOptionExists)
	{
		siOutOptionValue = atol(sStringValue.c_str());
	}

	bool bResult = bOptionExists;
	return bResult;
}

bool CConfigStorage::GetOptionInt(ECONFIGOPTION coConfigOption, ushort &usOutOptionValue) const
{
	bool bResult = false;
	string sStringValue;

	if (GetOptionString(coConfigOption, sStringValue))
	{
		usOutOptionValue = (ushort)atol(sStringValue.c_str());
		bResult = true;
	}

	return bResult;
}


void CConfigStorage::Dump()
{
	COptionsStorage &osOptionsStorage = GetOptionsStorage();

	COptionsStorage::const_iterator itStorage = osOptionsStorage.begin();
	for (; itStorage != osOptionsStorage.end(); ++itStorage)
	{
		cout << itStorage->first << "\t=>\t" << itStorage->second.c_str() << endl;
	}
}
