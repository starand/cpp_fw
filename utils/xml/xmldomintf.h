#ifndef __XMLDOMINTF_H__
#define __XMLDOMINTF_H__

#include <assert.h>
#include "XMLParserLite.h"

//forward class declarations
class CXMLDOMNodeList;
class CXMLDOMAttr;
class CXMLDOMNode;
class CXMLDOMElement;
class CXMLDOMDocument;
class CXMLDOMCharacterData;
class CXMLDOMText;
class CXMLDOMCDATASection;
class CXMLDOMComment;

typedef vector<CXMLDOMAttr> CXMLDOMAttributesList;


// Containers


/*

	CXMLDOMBaseImpl

*/


class CXMLDOMBaseImpl
{
public:
	CXMLDOMBaseImpl(): m_pImpl(NULL) {}
	CXMLDOMBaseImpl(const CXMLDOMBaseImpl &biOther);
	~CXMLDOMBaseImpl();

protected:
    CXMLDOMBaseImpl(XMLDOM_Base *pImpl): m_pImpl(pImpl) {}

public:
    CXMLDOMBaseImpl &operator =(const CXMLDOMBaseImpl &biOther);
	CXMLDOMBaseImpl &operator =(const void *pOfNULL);

    bool operator ==(const CXMLDOMBaseImpl &biOther) const { return operator ==(biOther.m_pImpl); }
    bool operator ==(const XMLDOM_Base *pOtherImpl) const { return m_pImpl == pOtherImpl; }
    bool operator !=(const CXMLDOMBaseImpl &biOther) const { return !operator ==(biOther); }
    bool operator !=(const XMLDOM_Base *pOtherImpl) const { return !operator ==(pOtherImpl); }

	bool isNull() const { return GetImpl() == NULL; }
	void setNull() { operator =(NULL); }
	void setNULL() { setNull(); }

protected:
	XMLDOM_Base *GetImpl() const { return m_pImpl; }

private:
    XMLDOM_Base			*m_pImpl;
};


/*

	CXMLDOMNodeList

*/

typedef CXMLDOMBaseImpl CXMLDOMNodeList_Parent;
class  CXMLDOMNodeList:
	public CXMLDOMNodeList_Parent
{
public:
	friend class CXMLDOMNode;
	friend class CXMLDOMElement;
	
public:
    CXMLDOMNodeList() {}
    CXMLDOMNodeList(const CXMLDOMNodeList &nlOther): CXMLDOMNodeList_Parent(nlOther) {}

protected:
    CXMLDOMNodeList(XMLDOM_NodeList *pImpl): CXMLDOMNodeList_Parent(pImpl) {}

public:
    bool operator ==(const CXMLDOMNodeList &nlOther) const { return CXMLDOMNodeList_Parent::operator ==(nlOther); }
    bool operator ==(const XMLDOM_NodeList *pOtherImpl) const { return CXMLDOMNodeList_Parent::operator ==(pOtherImpl); }
    bool operator !=(const CXMLDOMNodeList &nlOther) const { return CXMLDOMNodeList_Parent::operator !=(nlOther); }
    bool operator !=(const XMLDOM_NodeList *pOtherImpl) const { return CXMLDOMNodeList_Parent::operator !=(pOtherImpl); }

public:

    unsigned int getLength() const;
	CXMLDOMNode item(unsigned int index) const;

	void erase(unsigned int index);
	CXMLDOMNodeList cloneNodes(bool deep=true);	  

protected:
	XMLDOM_NodeList *GetImpl() const { return static_cast<XMLDOM_NodeList *>(CXMLDOMNodeList_Parent::GetImpl()); }

};



/*

	CXMLDOMNode

*/


typedef CXMLDOMBaseImpl CXMLDOMNode_Parent;
class CXMLDOMNode:
	public CXMLDOMNode_Parent
{
public:
	friend class CXMLDOMNodeList;
	friend class CXMLDOMDocument;
	
public:
    CXMLDOMNode() {}
    CXMLDOMNode(const CXMLDOMNode &nOther): CXMLDOMNode_Parent(nOther) {}
	~CXMLDOMNode() {}

protected:
    CXMLDOMNode(XMLDOM_Node *pImpl): CXMLDOMNode_Parent(pImpl) {}

public:
	bool operator !() const { return isNull(); }

    bool operator ==(const CXMLDOMNode &nOther) const { return CXMLDOMNode_Parent::operator ==(nOther); }
    bool operator ==(const XMLDOM_Node *pOtherImpl) const { return CXMLDOMNode_Parent::operator ==(pOtherImpl); }
    bool operator !=(const CXMLDOMNode &nOther) const { return CXMLDOMNode_Parent::operator !=(nOther); }
    bool operator !=(const XMLDOM_Node *pOtherImpl) const { return CXMLDOMNode_Parent::operator !=(pOtherImpl); }

public:
    int getNodeType() const;
	
    CXMLDOMElement getParentNode() const;
    CXMLDOMDocument getOwnerDocument() const;
	
    CXMLDOMNode cloneNode(bool deep) const;

    string getNodeName() const;
    string getNodeValue() const;
	void setNodeName(const string &nodeName);
    void setNodeValue(const string &nodeValue);

	bool hasChildNodes() const;
    CXMLDOMNodeList getChildNodes() const;
	void removeChildNodes();

	CXMLDOMNode getFirstChild() const;
    CXMLDOMNode getLastChild() const;
    CXMLDOMNode getPreviousSibling() const;
    CXMLDOMNode getNextSibling() const;
	int getIndexInParent() const; 
	void removeFromParent();

	CXMLDOMNode insertBefore(const CXMLDOMNode &newChild, const CXMLDOMNode &refChild);
    CXMLDOMNode replaceChild(const CXMLDOMNode &newChild, const CXMLDOMNode &oldChild);
    CXMLDOMNode removeChild(const CXMLDOMNode &oldChild);
    CXMLDOMNode appendChild(const CXMLDOMNode &newChild);
    
	void clear();
	void Clear() { clear(); } // For backward compatibility
	void normalize();
	
	string get_xml(EncodingCharset nCharset = ECS_NONE) const;
	void get_xml(string & sXML, EncodingCharset nCharset = ECS_NONE) const;
	string get_title() const;
	string get_text() const;
	void put_text(const string &text);

	CXMLDOMNodeList selectNodesByPattern(const CXMLDOMNode &PatternNode, bool deep = false);
	CXMLDOMNodeList selectNodes(const string &patternString, bool bCaseInsensitive = false) const;
	CXMLDOMNode selectSingleNode(const string &patternString, bool bCaseInsensitive = false) const;


protected:
	XMLDOM_Node *GetImpl() const { return static_cast<XMLDOM_Node *>(CXMLDOMNode_Parent::GetImpl()); }
};


typedef CXMLDOMNode CXMLDOMElement_Parent;
class CXMLDOMElement:
	public CXMLDOMElement_Parent
{
public:
	friend class CXMLDOMNode;
	friend class CXMLDOMDocument;
	
public:
	CXMLDOMElement() {}
	CXMLDOMElement(const CXMLDOMNode &eOther): CXMLDOMElement_Parent(eOther) { assert(eOther.isNull() || eOther.getNodeType() == NODE_ELEMENT); }
	
protected:
	CXMLDOMElement(XMLDOM_Element *pImpl): CXMLDOMElement_Parent(pImpl) {}
	
public:
	string getTagName() const;
	string getNamespace() const;
	bool getNamespaceByURI(const string& sURI, string& sNamespace) const;
	
	string getAttribute(const string &name) const;
	string getAttribute(const string &name, bool &bRes) const;
	void setAttribute(const string &name, const string &value);
	void removeAttribute(const string &name);
	
	CXMLDOMAttr getAttributeNode(const string &name) const;
	CXMLDOMAttr setAttributeNode(const CXMLDOMAttr &newAttr);
	CXMLDOMAttr removeAttributeNode(const CXMLDOMAttr &oldAttr);
	
	bool getAttributesList(CXMLDOMAttributesList &attrList) const;
	
	CXMLDOMNodeList getElementsByTagName(const string &name) const;
	
protected:
	XMLDOM_Element *GetImpl() const { return static_cast<XMLDOM_Element *>(CXMLDOMElement_Parent::GetImpl()); }
};

typedef CXMLDOMElement CXMLDOMDocument_Parent;
class CXMLDOMDocument:
	public CXMLDOMDocument_Parent
{
public:
	friend class CXMLDOMNode;
	
public:
	CXMLDOMDocument() {}
	CXMLDOMDocument(const CXMLDOMDocument &dOther): CXMLDOMDocument_Parent(dOther) {}
	
protected:
	CXMLDOMDocument(XMLDOM_Document *pImpl): CXMLDOMDocument_Parent(pImpl) {}
	
public:
	static CXMLDOMDocument createDocument();
	
	CXMLDOMElement createElement(const string &tagName, const string &tagNamespace = "") const;
	CXMLDOMText createTextNode(const string &data) const;
	CXMLDOMComment createComment(const string &data) const;
	CXMLDOMCDATASection createCDATASection(const string &data) const;
	CXMLDOMAttr createAttribute(const string &name) const;
	
	CXMLDOMElement getDocumentElement() const;

	EncodingCharset getCharset() const;
	void setCharset(EncodingCharset nCharset);
	
public:
	CXMLDOMNode StringToNode(const string &from);
	
	bool loadXML (const string &strxml);
	bool loadXMLEx(const string &strxml, string &strerr);
	bool loadXMLFiltered (const string &strxml, XMLDOM_FilterProc fpFilter, void *pvContext);
	
protected:
	XMLDOM_Document *GetImpl() const { return static_cast<XMLDOM_Document *>(CXMLDOMDocument_Parent::GetImpl()); }
};


typedef CXMLDOMNode CXMLDOMAttr_Parent;
class CXMLDOMAttr: 
	public CXMLDOMAttr_Parent
{
public:
	friend class CXMLDOMElement;
	friend class CXMLDOMDocument;

public:
    CXMLDOMAttr() {}
    CXMLDOMAttr(const CXMLDOMAttr &aOther): CXMLDOMAttr_Parent(aOther) {}

protected:
	CXMLDOMAttr(XMLDOM_Attr *pImpl): CXMLDOMAttr_Parent(pImpl) {}

public:
    string getName() const;
    string getValue() const;
    void setValue(const string &value);

    CXMLDOMElement getOwnerElement() const { return CXMLDOMAttr_Parent::getParentNode(); }

protected:
	XMLDOM_Attr *GetImpl() const { return static_cast<XMLDOM_Attr *>(CXMLDOMAttr_Parent::GetImpl()); }
};


typedef CXMLDOMNode CXMLDOMCharacterData_Parent;
class CXMLDOMCharacterData:
	public CXMLDOMCharacterData_Parent
{
public:
    CXMLDOMCharacterData() {}
    CXMLDOMCharacterData(const CXMLDOMCharacterData &cdOther): CXMLDOMCharacterData_Parent(cdOther) {}
	
protected:
	CXMLDOMCharacterData(XMLDOM_CharacterData *pImpl): CXMLDOMCharacterData_Parent(pImpl) {}

public:
	string getData() const;
	void setData(const string &data);

	string substringData(unsigned int offset, unsigned int count) const;

	void appendData(const string &arg);
	void insertData(unsigned int offset, const  string &arg);
	void deleteData(unsigned int offset, unsigned int count);
	void replaceData(unsigned int offset, unsigned int count, const string &arg);

protected:
	XMLDOM_CharacterData *GetImpl() const { return static_cast<XMLDOM_CharacterData *>(CXMLDOMAttr_Parent::GetImpl()); }
};


typedef CXMLDOMCharacterData CXMLDOMComment_Parent;
class CXMLDOMComment:
	public CXMLDOMComment_Parent
{
public:
	friend class CXMLDOMDocument;

public:
    CXMLDOMComment() {}
    CXMLDOMComment(const CXMLDOMComment &cOther): CXMLDOMComment_Parent(cOther) {}

protected:
	CXMLDOMComment(XMLDOM_Comment *pImpl): CXMLDOMComment_Parent(pImpl) {}
};


typedef CXMLDOMCharacterData CXMLDOMText_Parent;
class CXMLDOMText:
	public CXMLDOMText_Parent
{
public:
	friend class CXMLDOMDocument;
	
public:
    CXMLDOMText() {}
    CXMLDOMText(const CXMLDOMText &tOther): CXMLDOMText_Parent(tOther) {}
	
protected:
	CXMLDOMText(XMLDOM_Text *pImpl): CXMLDOMText_Parent(pImpl) {}
};


typedef CXMLDOMCharacterData CXMLDOMCDATASection_Parent;
class CXMLDOMCDATASection:
	public CXMLDOMCDATASection_Parent
{
public:
	friend class CXMLDOMDocument;
	
public:
	CXMLDOMCDATASection() {}
	CXMLDOMCDATASection(const CXMLDOMCDATASection &cdOther): CXMLDOMCDATASection_Parent(cdOther) {}

protected:
	CXMLDOMCDATASection(XMLDOM_CDATASection *pImpl): CXMLDOMCDATASection_Parent(pImpl) {}
};

string NormalizeXMLDOMDocument( CXMLDOMNode xmlNode );

#endif //__XMLDOMINTF_H__
