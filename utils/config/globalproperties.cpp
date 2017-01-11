#include "StdAfx.h"
#include "globalproperties.h"
#include "strutils.h"
#include <vector>

const char szInitPropDelimiter[] = "\n";
const char szNameValueDelimiter[] = "=";

///////////////////////////////////////////////////////////////////////////////////////
//	CGlobalProperties implementation

CGlobalProperties::CGlobalProperties( const char* szInitializeProperties /*= ""*/ )
{
	if( szInitializeProperties && strlen(szInitializeProperties) )
	{
		vector<string> vsParseResult;
		StrUtils::Explode( szInitializeProperties, szInitPropDelimiter, vsParseResult );

		size_t nResultSize = vsParseResult.size();
		for( size_t i = 0; i < nResultSize; ++i )
		{
			vector<string> vsTagValue;
			StrUtils::Explode( vsParseResult[i].c_str(), szNameValueDelimiter, vsTagValue );

			if( vsTagValue.size() < 2 ) 
			{
				continue;
			}

			SetProperty( vsTagValue[0].c_str(), vsTagValue[1].c_str() );
		}
	}
}

void CGlobalProperties::SetProperty(GLOBAL_PROPERTY_TYPE gptProperty, const char *szValue)
{
	if(gptProperty == GPROP_UNKNOWN) return;

	string &sPropertyValue = operator[]( gptProperty );
	if( strcmp(sPropertyValue.c_str(), szValue) != 0 )
	{
		sPropertyValue = szValue;
	}
}

void CGlobalProperties::SetProperty(const char *szProperty, const char *szValue)
{
	SetProperty( DecodeGlobalPropertyType(szProperty), szValue );
}

bool CGlobalProperties::GetProperty(GLOBAL_PROPERTY_TYPE gptProperty, string &sValue) const
{
	bool bResult = FALSE;

	do 
	{
		sValue.clear();

		if (gptProperty == GPROP_UNKNOWN) break;

		const_iterator it = find(gptProperty);
		if( it != end() )
		{
			sValue = (*it).second;
			bResult = TRUE;
		}
	}
	while(FALSE);

	return bResult;
}

bool CGlobalProperties::GetProperty(const char *szProperty, string &sValue) const
{
	return GetProperty( DecodeGlobalPropertyType(szProperty), sValue);
}

const char* CGlobalProperties::GetProperty(GLOBAL_PROPERTY_TYPE gptProperty) const
{
	const char* szValue = NULL;
	
	const_iterator it = find(gptProperty);
	if( it != end() )
	{
		szValue = (*it).second.c_str();
	}

	return szValue;
}

// Encode/decode for global properties
const char* CGlobalProperties::EncodeGlobalPropertyType(GLOBAL_PROPERTY_TYPE gptPropertyType)
{
	const char *szValue = NULL;

	if( gptPropertyType < GPROP__MAX )
	{
		szValue = aszGlobalProperties[gptPropertyType];
	}

	return szValue;
}

GLOBAL_PROPERTY_TYPE CGlobalProperties::DecodeGlobalPropertyType(const char *szPropertyType)
{
	GLOBAL_PROPERTY_TYPE gptResult = GPROP_UNKNOWN;

	for(size_t gpIdx = GPROP__MIN; gpIdx < GPROP__MAX; ++gpIdx)
	{
		if( strcmp(aszGlobalProperties[gpIdx], szPropertyType) == 0 )
		{
			gptResult = (GLOBAL_PROPERTY_TYPE)gpIdx;
			break;
		}
	}

	return gptResult;
}
