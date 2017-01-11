#ifndef __H_GLOBALPROPERTIES__
#define __H_GLOBALPROPERTIES__

#include "globalproperties_enums.h"
#include <map>

typedef map<GLOBAL_PROPERTY_TYPE, string> CGlobalPropertiesMap_Parent;

class CGlobalProperties 
	: public CGlobalPropertiesMap_Parent
{
public:
	CGlobalProperties( const char* szInitializeProperties = "" );

	void SetProperty(GLOBAL_PROPERTY_TYPE gptProperty, const char *szValue);
	void SetProperty(const char *szProperty, const char *szValue);

	bool GetProperty(GLOBAL_PROPERTY_TYPE gptProperty, string &sValue) const;
	bool GetProperty(const char *szProperty, string &sValue) const;
	const char* GetProperty(GLOBAL_PROPERTY_TYPE gptProperty) const;

public:
	static const char* EncodeGlobalPropertyType(GLOBAL_PROPERTY_TYPE gptPropertyType);
	static GLOBAL_PROPERTY_TYPE DecodeGlobalPropertyType(const char *szPropertyType);
};

#endif // __H_GLOBALPROPERTIES__
