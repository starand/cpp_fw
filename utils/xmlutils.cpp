#include "StdAfx.h"
#include "xmlutils.h"

#include "xmldomintf.h"
#include "fileutils.h"

namespace CXMLUtils
{
	bool LoadXMLFile(const string &sXMLFileName, CXMLDOMElement &xdeOutPropertyElement)
	{
		bool bResult = false;

		do 
		{
			string sXMLFileContent;
			if (!FileUtils::GetFileContent(sXMLFileName.c_str(), sXMLFileContent))
			{
				break;
			}

			CXMLDOMDocument &xddXMLDocument = CXMLDOMDocument::createDocument();
			ASSERTE(!xddXMLDocument.isNull());

			if (!xddXMLDocument.loadXML(sXMLFileContent))
			{
				break;
			}

			xdeOutPropertyElement = xddXMLDocument.getDocumentElement();
			bResult = true;
		}
		while (false);

		return bResult;
	}

};

