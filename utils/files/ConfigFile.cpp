#include "StdAfx.h"
#include "ConfigFile.h"
#include <logger.h>
#include <asserts.h>
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include "strutils.h"

#define COMMENT_BEGIN	';'
#define EQUAL_SIGN		'='

#define LEFT_BRACKET	'['
#define RIGHT_BRACKET	']'


const char g_szEFileOpen[] = "Can\'t open file : %s";
const char g_szSpaceChars[] = " \t";


////////////////////////////////////////////////////
// CIniSettings implementation

CConfigFile::CConfigFile() { }

CConfigFile::CConfigFile(const char* szFileName)
{
	LoadFromFile( szFileName );
}


CConfigFile::~CConfigFile()
{
	Clear();
}


bool CConfigFile::LoadFromFile(const string& sFileName, bool bHideError /*= false*/)
{
	bool bResult = false;

	do
	{
		ifstream fin( sFileName.c_str() );
		if (!fin.is_open())
		{
			if (bHideError)
			{
				LOG_ERROR(g_szEFileOpen, sFileName.c_str());
			}
			break;
		}

		string sFileLine, sTag, sValue;
		CSectionData* pSubMap = NULL;
		while( !fin.eof() && fin.good() )
		{
			getline( fin, sFileLine );
			if( sFileLine.empty()  || sFileLine[0] == COMMENT_BEGIN ) continue;
			if( sFileLine[0] == LEFT_BRACKET )
			{
				//StrUtils::RemoveBrackets( sFileLine, '[', ']' );
				RemoveBrackets(sFileLine);

				CIniSettingStorage::const_iterator itMap = m_issStorage.find( sFileLine );
				if( itMap == m_issStorage.end() ) 
				{
					pSubMap = new CSectionData;
					m_issStorage.insert( CIniSettingStorage::value_type(sFileLine, pSubMap) );
				}
				else
				{
					pSubMap = itMap->second;
				}
				continue;
			}

			if (pSubMap)
			{
				ParseKeyValue(sFileLine, sTag, sValue);

				if( !sTag.empty() )
				{
					(*pSubMap)[ sTag ] = sValue;
				}
			}		
		}

		bResult = true;
	}
	while(false);

	return bResult;
}

void CConfigFile::Clear()
{
	for( CIniSettingStorage::iterator itMap = m_issStorage.begin(); itMap != m_issStorage.end(); itMap++ )
	{
		delete itMap->second;
	}

	m_issStorage.clear();
}


void CConfigFile::Debug()
{
	for( CIniSettingStorage::iterator itMap = m_issStorage.begin(); itMap != m_issStorage.end(); ++itMap )
	{
		CSectionData& sdSectionData = *(itMap->second);
		std::cout << LEFT_BRACKET << itMap->first.c_str() << RIGHT_BRACKET << std::endl;

		for( CSectionData::iterator itSub = sdSectionData.begin();
			itSub != sdSectionData.end(); ++itSub )
		{
			std::cout << itSub->first.c_str() << " = " << itSub->second.c_str() << std::endl;
		}
	}
}

bool CConfigFile::GetOptionValue( const string& sSection, const string& sParam, string& sValue )
{
	bool bResult = false;

	do
	{
		CIniSettingStorage::iterator itMap = m_issStorage.find( sSection );
		if( itMap == m_issStorage.end() )
		{
		    LOG_ERROR("Unable to find section %s", sSection.c_str());
		    break;
		}

		CSectionData& sdSectionData = *(itMap->second);
		CSectionData::iterator itSubMup = sdSectionData.find( sParam );
		if( itSubMup == sdSectionData.end() )
		{
		    LOG_ERROR("Unable to search key %s", sParam.c_str());
		    break;
		}

		// if value exists
		sValue.assign( itSubMup->second.c_str() );
		bResult = true;
	}
	while(false);

	return bResult;
}

void CConfigFile::GetOptionValue( const string& sSection, const string& sParam, string& sValue, const char* szDefaultValue )
{
	if( !GetOptionValue(sSection, sParam, sValue) ) sValue = ( szDefaultValue ? szDefaultValue : "" );
}


bool CConfigFile::GetOptionValue( const string& sSection, const string& sParam, int& nValue )
{
	bool bResult = false;
	
	string sValue;
	if (true == (bResult = GetOptionValue(sSection, sParam, sValue)))
	{
		nValue = atoi( sValue.c_str() );
	}

	return bResult;
}

void CConfigFile::GetOptionValue(const string& sSection, const string& sParam, int& nValue, int nDefValue)
{
	if (!GetOptionValue(sSection, sParam, nValue))
	{
		nValue = nDefValue;
	}
}


bool CConfigFile::GetOptionValue(const string& sSection, const string& sParam, ushort& nValue)
{
	bool bResult = false;
	
	string sValue;
	if (true == (bResult = GetOptionValue(sSection, sParam, sValue)))
	{
		nValue = atoi( sValue.c_str() );
	}

	return bResult;
}

void CConfigFile::GetOptionValue(const string& sSection, const string& sParam, ushort& nValue, ushort usDefValue)
{
	if (!GetOptionValue(sSection, sParam, nValue))
	{
		nValue = usDefValue;
	}
}


void CConfigFile::SetOptionValue(const string& sSection, const string& sParam, const string& sValue)
{
	CIniSettingStorage::iterator itMap = m_issStorage.find(sSection);
	if (itMap == m_issStorage.end())
	{
		 CSectionData *psdSectionData = new (nothrow) CSectionData();
		 ASSERTE(psdSectionData);

		const pair<CIniSettingStorage::iterator, bool>& insResult = m_issStorage.insert(CIniSettingStorage::value_type(sSection, psdSectionData));
		itMap = insResult.first;
	}

	CSectionData& sdSectionData = *(itMap->second);
	CSectionData::iterator itSubMup = sdSectionData.find(sParam);

	if (itSubMup == sdSectionData.end())
	{
		sdSectionData.insert(CSectionData::value_type(sParam, sValue));
	}
	else
	{
		itSubMup->second = sValue;
	}
}


void CConfigFile::RemoveBrackets(string& sVarValue) const
{
	sizeint length = sVarValue.length();
	if (length > 2)
	{
	    if (sVarValue[length-1] == ']') sVarValue.erase(length-1);
	    if (sVarValue[0] == '[') sVarValue.erase(0, 1);
	}
}

void CConfigFile::ParseKeyValue(const string& sInputValue, string& sOutKey, string& sOutValue) const
{
    ASSERT(!sInputValue.empty());
    size_t equalPos = sInputValue.find('=');

    if (equalPos != string::npos)
    {
        sOutKey = sInputValue.substr(0, equalPos);
        sOutValue = sInputValue.substr(equalPos + 1);
        StrUtils::Trim(sOutKey);
        StrUtils::Trim(sOutValue);
    }
}
