#ifndef __XMLUTILS_H_INCLUDED
#define __XMLUTILS_H_INCLUDED


class CXMLDOMElement;

namespace CXMLUtils
{
	bool LoadXMLFile(const string &sXMLFileName, CXMLDOMElement &xdeOutPropertyElement);

};

#endif // __XMLUTILS_H_INCLUDED
