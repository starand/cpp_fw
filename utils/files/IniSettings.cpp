#include "StdAfx.h"
#include "IniSettings.h"
#include "logerror.h"
#include <fstream>

#define COMMENT_BEGIN ';'
#define SECTION_BEGIN '['

const char szEFileOpen[] = "Can\'t open file : %s";

////////////////////////////////////////////////////
// CIniSettings implementation

CIniSettings::CIniSettings() { }

CIniSettings::CIniSettings( const char* szFileName )
{
	LoadFromFile( szFileName );
}

CIniSettings::~CIniSettings()
{
	Clear();
}

void CIniSettings::Clear()
{
	for( CIniSettingStorage::iterator itMap = m_issStorage.begin(); itMap != m_issStorage.end(); itMap++ )
	{
		delete itMap->second;
	}

	m_issStorage.clear();
}

bool CIniSettings::LoadFromFile(const string& sFileName, bool bHideError /*= false*/)
{
	bool bResult = false;

	do
	{
		ifstream fin( sFileName.c_str() );
		if (!fin.is_open())
		{
			if (bHideError)
			{
				LOG_ERROR2(szEFileOpen, sFileName.c_str());
			}
			break;
		}

		string sFileLine, sTag, sValue;
		CSectionData* pSubMap = NULL;
		while( !fin.eof() && fin.good() )
		{
			getline( fin, sFileLine );
			if( sFileLine.empty()  || sFileLine[0] == COMMENT_BEGIN ) continue;
			if( sFileLine[0] == SECTION_BEGIN )
			{
				StrUtils::RemoveBrackets( sFileLine, '[', ']' );

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

			if( pSubMap )
			{
				StrUtils::ParseNameValuePair( sFileLine.c_str(), sTag, sValue );
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

void CIniSettings::Debug()
{
	for( CIniSettingStorage::iterator itMap = m_issStorage.begin(); itMap != m_issStorage.end(); ++itMap )
	{
		CSectionData& sdSectionData = *(itMap->second);
		cout << "[" << itMap->first.c_str() << "]" << endl;

		for( CSectionData::iterator itSub = sdSectionData.begin();
			itSub != sdSectionData.end(); ++itSub )
		{
			cout << itSub->first.c_str() << " = " << itSub->second.c_str() << endl;
		}
	}
}

bool CIniSettings::GetSectionValue( const string& sSection, const string& sParam, string& sValue )
{
	bool bResult = false;

	do
	{
		CIniSettingStorage::iterator itMap = m_issStorage.find( sSection );
		if( itMap == m_issStorage.end() ) break;

		CSectionData& sdSectionData = *(itMap->second);
		CSectionData::iterator itSubMup = sdSectionData.find( sParam );
		if( itSubMup == sdSectionData.end() ) break;

		// if value exists
		sValue.assign( itSubMup->second.c_str() );
		bResult = true;
	}
	while(false);

	return bResult;
}

void CIniSettings::GetSectionValue( const string& sSection, const string& sParam, string& sValue, const char* szDefaultValue )
{
	if( !GetSectionValue(sSection, sParam, sValue) ) sValue = ( szDefaultValue ? szDefaultValue : "" );
}

bool CIniSettings::GetSectionValue( const string& sSection, const string& sParam, int& nValue )
{
	bool bResult = false;
	
	string sValue;
	if( bResult = GetSectionValue(sSection, sParam, sValue) ) {
		nValue = atoi( sValue.c_str() );
	}

	return bResult;
}

void CIniSettings::GetSectionValue(const string& sSection, const string& sParam, int& nValue, int nDefValue)
{
	if (!GetSectionValue(sSection, sParam, nValue)) nValue = nDefValue;
}


bool CIniSettings::GetSectionValue(const string& sSection, const string& sParam, ushort& nValue)
{
	bool bResult = false;
	
	string sValue;
	if( bResult = GetSectionValue(sSection, sParam, sValue) ) {
		nValue = atoi( sValue.c_str() );
	}

	return bResult;
}

void CIniSettings::GetSectionValue(const string& sSection, const string& sParam, ushort& nValue, ushort usDefValue)
{
	if (!GetSectionValue(sSection, sParam, nValue))
	{
		nValue = usDefValue;
	}
}


void CIniSettings::SetSectionValue(const string& sSection, const string& sParam, const string& sValue)
{
	CIniSettingStorage::iterator itMap = m_issStorage.find(sSection);
	if (itMap == m_issStorage.end())
	{
		 CSectionData *psdSectionData = new (nothrow) CSectionData();
		 ASSERTE(psdSectionData);

		const auto &insResult = m_issStorage.insert(CIniSettingStorage::value_type(sSection, psdSectionData));
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


CSectionData &CIniSettings::operator[]( const string& sSectionName )
{
	CSectionData* pResult = NULL;

	CIniSettingStorage::iterator itMap = m_issStorage.find( sSectionName );
	if( m_issStorage.end() == itMap ) 
	{
		pResult = new CSectionData;
		m_issStorage.insert( CIniSettingStorage::value_type(sSectionName, pResult) );
	}
	else
	{
		pResult = itMap->second;
	}

	return *pResult;
}
