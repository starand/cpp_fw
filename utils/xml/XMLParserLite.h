#ifndef __XMLPARSERLITE__
#define __XMLPARSERLITE__

#include <assert.h>
#include "constants.h"

#ifndef snprintf
#	define snprintf _snprintf
#endif

#ifdef LINUX	// avoid MS VC++ precompiler bug
#include <map>
#include <vector>
#endif

enum DOMNodeType
{
	NODE_INVALID,
	NODE_ELEMENT,
	NODE_ATTRIBUTE,
	NODE_TEXT,
	NODE_CDATA_SECTION,
	NODE_COMMENT,
	NODE_DOCUMENT,

	NODE__MAX
};

enum EncodingCharset
{
	ECS_NONE,
	ECS_ASCII,
	ECS_UTF8
};

extern const char szXMLPparser_UTF8_BOM[];
extern const char szXMLPparser_UTF8_Encoding[];
extern const char szXMLPparser_UTF16_Encoding[];
extern const char szXMLPparser_AttrEncoding[];
extern const char szXMLPparser_AttrVersion[];

// Forward class declarations
class XMLDOM_Document;
class XMLDOM_Node;
class XMLDOM_Element;
class XMLDOM_Text;
class XMLDOM_Comment;
class XMLDOM_CDATASection;
class XMLDOM_Attr;
class XMLDOM_NodeList;

typedef bool (*XMLDOM_FilterProc)(const string &sNodeName, DOMNodeType ntNodeType, unsigned int nNodeLevel, void *pvContext);
#define DOM_LEVEL_ROOT	0U

class XMLDOM_Base 
{
public:
	friend class XMLDOM_InternalNodeStorage;

public:
	XMLDOM_Base();

protected:
	virtual ~XMLDOM_Base(); // Use Release()

public:
	virtual void AddRef();
	virtual void Release();

protected:
	void InternalAddRef();
	void InternalRelease();

	unsigned int GetRefCount() const { return m_ulRefCount; }

private:
	XMLDOM_Base &operator =(const XMLDOM_Base &) { /* Why would one need to copy xml implementation nodes??? */ return *this; }

private:
	unsigned int	m_ulRefCount;
};


typedef vector<XMLDOM_Node *> XMLDOM_InternalNodeStorage_Parent;
class XMLDOM_InternalNodeStorage:
	protected XMLDOM_InternalNodeStorage_Parent
{
public:
	friend class XMLDOM_Element;

	typedef XMLDOM_InternalNodeStorage_Parent::size_type size_type;
	enum
	{
		npos = ~((size_type)0)
	};
	
public:
	XMLDOM_InternalNodeStorage() {}
	XMLDOM_InternalNodeStorage(const XMLDOM_InternalNodeStorage &nsOther, size_type nBeginIndex=0, size_type nEndIndex=npos);
	/*not virtual*/ ~XMLDOM_InternalNodeStorage();

private:
	void CloneContents(const XMLDOM_InternalNodeStorage &nsOther, size_type nBeginIndex, size_type nEndIndex);
	void ClearContents();

protected:
	void SetItemsParent(XMLDOM_Element *peParentNode, size_type nBeginIndex=0, size_type nEndIndex=npos);
	void SetItemsOwnerDocument(XMLDOM_Document *pdNewDocument);

public:
	size_type GetLength() const { return size(); }
	XMLDOM_Node *GetUnsafeItem(size_type nItemIndex) const;
	
	void AddItem(XMLDOM_Node *pnNewItem);
	bool RemoveItem(size_type nItemIndex);
	bool ReplaceItem(size_type nItemIndex, XMLDOM_Node *pnNewItem);
	bool InsertItem(size_type nItemIndex, XMLDOM_Node *pnNewItem);

	void RemoveItems(size_type nBeginIndex=0, size_type nEndIndex=npos);

public:
	size_type GetItemIndex(XMLDOM_Node *pnItemOfInterest, size_type nBeginIndex=0, size_type nEndIndex=npos) const;
};


typedef XMLDOM_Base XMLDOM_NodeList_Parent;
typedef XMLDOM_InternalNodeStorage XMLDOM_NodeList_VectorParent;
class XMLDOM_NodeList:
	public XMLDOM_NodeList_Parent,
	protected XMLDOM_NodeList_VectorParent
{
public:
	typedef XMLDOM_NodeList_VectorParent::size_type size_type;
	enum
	{
		npos = XMLDOM_NodeList_VectorParent::npos
	};

public:
	XMLDOM_NodeList() {}
	XMLDOM_NodeList(const XMLDOM_NodeList &nlOther);

protected:
	virtual ~XMLDOM_NodeList(); // Use Release()

private:
	void AddRefList();
	void ReleaseList();

public:
	size_type GetLength() const { return XMLDOM_NodeList_VectorParent::GetLength(); }
	bool GetItem(size_type nItemIndex, XMLDOM_Node *&pnOutItem) const;

	void AddItem(XMLDOM_Node *pnItem);
	bool RemoveItem(size_type nItemIndex);

public:
	void FillFromStorage(XMLDOM_InternalNodeStorage *pnsOtherNodes, size_type nBeginIndex=0, size_type nEndIndex=npos);
};


typedef XMLDOM_Base XMLDOM_Node_Parent;
class XMLDOM_Node:
	public XMLDOM_Node_Parent
{
public:
	friend class XMLDOM_InternalNodeStorage;
	friend class XMLDOM_Element;
	friend class XMLDOM_Document;

public:
    XMLDOM_Node(XMLDOM_Document *pdOwnerDocument);

	typedef XMLDOM_InternalNodeStorage::size_type size_type;
	enum
	{
		npos = XMLDOM_InternalNodeStorage::npos
	};

protected:
    XMLDOM_Node(const XMLDOM_Node &nOther);
	virtual ~XMLDOM_Node(); // Use Release()
	
public:
	virtual void AddRef();
	virtual void Release();

protected:
	unsigned int GetExternalRefCount() const { return m_ulExternalRefCount; }

public:
	virtual void CloneNode(XMLDOM_Node *&pnNewNode, bool bRecursive) const = 0;
	virtual void SetOwnerDocument(XMLDOM_Document *pdNewDocument);

public:
	virtual const string &VirtualGetNodeName() const;
	virtual void VirtualGetNodeValue(string &sValue) const;
	virtual void VirtualSetNodeName(const string &sNodeName);
	virtual void VirtualSetNodeValue(const string &sNodeValue);
	
	virtual bool VirtualHasChildNodes() const;
	virtual void VirtualGetChildNodes(XMLDOM_NodeList *&pnlChildNodes);
	virtual void VirtualRemoveChildNodes();

	virtual bool VirtualGetFirstChildNode(XMLDOM_Node *&pnChildNode);
	virtual bool VirtualGetLastChildNode(XMLDOM_Node *&pnChildNode);
	virtual bool VirtualGetPreviousSiblingNode(XMLDOM_Node *&pnChildNode);
	virtual bool VirtualGetNextSiblingNode(XMLDOM_Node *&pnChildNode);
	virtual size_type VirtualGetIndexInParent() const;

	virtual void VirtualAppendChildNode(XMLDOM_Node *pnNewChild);
	virtual bool VirtualRemoveChildNode(XMLDOM_Node *pnOldChild);
	virtual bool VirtualReplaceChildNode(XMLDOM_Node *pnNewChild, XMLDOM_Node *pnOldChild);
	virtual bool VirtualInsertChildNode(XMLDOM_Node *pnNewChild, XMLDOM_Node *pnOldChild);

	virtual void VirtualSelectElementsByPattern(XMLDOM_Element *pePatternNode, XMLDOM_NodeList *&pnlOutResultNodes, bool bRecursive=false);
	virtual void VirtualSelectNodes(const string &sPatternString, XMLDOM_NodeList *&pnlOutResultList, bool bCaseInsensitive=false);
	virtual bool VirtualSelectSingleNode(const string &sPatternString, XMLDOM_Node *&pnOutResultNode, bool bCaseInsensitive=false);

protected:
	void SetParentNode(XMLDOM_Element *peParentNode) { m_peParentNode = peParentNode; }

public:
	void GetOwnerDocument(XMLDOM_Document *&pdOutOwnerDocument) const;
	bool GetParentNode(XMLDOM_Element *&peOutParentNode) const;

	virtual DOMNodeType GetNodeType() const = 0;
	virtual bool IsElementOrDescendant() const;

protected:
	void SetNodeName(const string &sName) { m_sNodeName = sName; }
	void SetNodeNamespace(const string &sNamespace) { m_sNodeNamespace = sNamespace; }
    const string &GetNodeName() const { return m_sNodeName; }
	const string &GetNodeNamespace() const { return m_sNodeNamespace; }

protected:
	XMLDOM_Document *GetUnsafeOwnerDocument() const { return m_pdOwnerDocument; }
	XMLDOM_Element *GetUnsafeParentNode() const { return m_peParentNode; }

public:
	size_type GetIndexInParent() const;
	void RemoveFromParent();

	bool GetPreviousSiblingNode(XMLDOM_Node *&pnOutPreviousNode) const;
    bool GetNextSiblingNode(XMLDOM_Node *&pnOutNextNode) const;
	
public:
	virtual void GetTitle(string &sNodeTitle) = 0;
	virtual void GetText(string &sNodeText) = 0;
	virtual void PutText(const string &sNodeText) = 0; 
	virtual void Clear() = 0;

	void GetXML(string &sXMLString, EncodingCharset nCharset);
	
protected:
    virtual string::size_type InternalGetXML(string &sBuffer, EncodingCharset nCharset) const = 0;

private:
    unsigned int		m_ulExternalRefCount;

	XMLDOM_Document		*m_pdOwnerDocument;
	XMLDOM_Element		*m_peParentNode;
	
    string				m_sNodeName;
	string				m_sNodeNamespace;
};


typedef XMLDOM_Node XMLDOM_CharacterData_Parent;
class XMLDOM_CharacterData: 
	public XMLDOM_CharacterData_Parent
{
public:
	XMLDOM_CharacterData(XMLDOM_Document *pdOwnerDocument, const string &sData);

protected:
	XMLDOM_CharacterData(const XMLDOM_CharacterData &cdOther);
	virtual ~XMLDOM_CharacterData(); // Use Release()

public:
	virtual void VirtualGetNodeValue(string &sValue) const;
	virtual void VirtualSetNodeValue(const string &sNodeValue);

public:
	const string &GetData() const { return GetNodeName(); }
	void SetData(const string &sData) { SetNodeName(sData); }
	
	void AppendData(const string &sNewData);
	void InsertData(string::size_type nOffset, const  string &sNewData);
	void DeleteData(string::size_type nOffset, string::size_type nCount);
	void ReplaceData(string::size_type nOffset, string::size_type nCount, const string &sNewData);

public:
	virtual void GetTitle(string &sNodeTitle);
	virtual void GetText(string &sNodeText);
	virtual void PutText(const string &sNodeText); 
	virtual void Clear();
};


typedef XMLDOM_Node XMLDOM_Attr_Parent;
class XMLDOM_Attr:
	public XMLDOM_Attr_Parent
{
public:
	friend class XMLDOM_Element;
	friend class XMLDOM_Document;
	
public:
	XMLDOM_Attr(XMLDOM_Document *pdOwnerDocument, const string &sName);

protected:
	XMLDOM_Attr(const XMLDOM_Attr &aOther);
	virtual ~XMLDOM_Attr(); // Use Release()

public:
	virtual void CloneNode(XMLDOM_Node *&pnNewNode, bool bRecursive) const;

public:
	virtual const string &VirtualGetNodeName() const;
	virtual void VirtualGetNodeValue(string &sValue) const;
	virtual void VirtualSetNodeName(const string &sNodeValue);
	virtual void VirtualSetNodeValue(const string &sNodeValue);

	virtual bool VirtualGetPreviousSiblingNode(XMLDOM_Node *&pnChildNode);
	virtual bool VirtualGetNextSiblingNode(XMLDOM_Node *&pnChildNode);
	virtual size_type VirtualGetIndexInParent() const;

public:
    virtual DOMNodeType GetNodeType() const;

	const string &GetName() const { return GetNodeName(); }
	const string &GetValue() const { return m_sValue; }
	void SetName(const string &sName) { SetNodeName(sName); }
	void SetValue(const string &sValue) { m_sValue = sValue; }

public:
	virtual void GetTitle(string &sNodeTitle);
	virtual void GetText(string &sNodeText);
	virtual void PutText(const string &sNodeText); 
	virtual void Clear();

protected:
    virtual string::size_type InternalGetXML(string &sBuffer, EncodingCharset nCharset) const;
	
private:
	string				m_sValue;
};


typedef XMLDOM_CharacterData XMLDOM_Text_Parent;
class XMLDOM_Text:
	public XMLDOM_Text_Parent
{
public:
	XMLDOM_Text(XMLDOM_Document *pdOwnerDocument, const string &sContent);

protected:
	XMLDOM_Text(const XMLDOM_Text &tOther);
	virtual ~XMLDOM_Text(); // Use Release()

public:
	virtual void CloneNode(XMLDOM_Node *&pnNewNode, bool bRecursive) const;

public:
    virtual DOMNodeType GetNodeType() const;

protected:
    virtual string::size_type InternalGetXML(string &sBuffer, EncodingCharset nCharset) const;
};


typedef XMLDOM_CharacterData XMLDOM_Comment_Parent;
class XMLDOM_Comment:
	public XMLDOM_Comment_Parent
{
public:
	XMLDOM_Comment(XMLDOM_Document *pdOwnerDocument, const string &sContent);

protected:
	XMLDOM_Comment(const XMLDOM_Comment &cOther);
	virtual ~XMLDOM_Comment(); // Use Release()

public:
	virtual void CloneNode(XMLDOM_Node *&pnNewNode, bool bRecursive) const;

public:
    virtual DOMNodeType GetNodeType() const;

protected:
    virtual string::size_type InternalGetXML(string &sBuffer, EncodingCharset nCharset) const;
};


typedef XMLDOM_CharacterData XMLDOM_CDATASection_Parent;
class XMLDOM_CDATASection:
	public XMLDOM_CDATASection_Parent
{
public:
	XMLDOM_CDATASection(XMLDOM_Document *pdOwnerDocument, const string &sContent);
	
protected:
	XMLDOM_CDATASection(const XMLDOM_CDATASection &cdOther);
	virtual ~XMLDOM_CDATASection(); // Use Release()

public:
	virtual void CloneNode(XMLDOM_Node *&pnNewNode, bool bRecursive) const;

public:
	virtual DOMNodeType GetNodeType() const;

protected:
    virtual string::size_type InternalGetXML(string &sBuffer, EncodingCharset nCharset) const;
};


typedef XMLDOM_Node XMLDOM_Element_Parent;
class XMLDOM_Element:
	public XMLDOM_Element_Parent
{
public:
    XMLDOM_Element(XMLDOM_Document *pdOwnerDocument, const string &sElementName, const string &sElementNamespace = "");

protected:
	XMLDOM_Element(const XMLDOM_Element &eOther, bool bRecursive=false);
    virtual ~XMLDOM_Element(); // Use Release()

public:
	virtual void CloneNode(XMLDOM_Node *&pnNewNode, bool bRecursive) const;
	virtual void SetOwnerDocument(XMLDOM_Document *pdNewDocument);

public:
	virtual const string &VirtualGetNodeName() const;
	virtual void VirtualGetNodeValue(string &sValue) const;
	virtual void VirtualSetNodeName(const string &sNodeValue);
	virtual void VirtualSetNodeValue(const string &sNodeValue);

	virtual bool VirtualHasChildNodes() const;
	virtual void VirtualGetChildNodes(XMLDOM_NodeList *&pnlChildNodes);
	virtual void VirtualRemoveChildNodes();

	virtual bool VirtualGetFirstChildNode(XMLDOM_Node *&pnChildNode);
	virtual bool VirtualGetLastChildNode(XMLDOM_Node *&pnChildNode);

	virtual void VirtualAppendChildNode(XMLDOM_Node *pnNewChild);
	virtual bool VirtualRemoveChildNode(XMLDOM_Node *pnOldChild);
	virtual bool VirtualReplaceChildNode(XMLDOM_Node *pnNewChild, XMLDOM_Node *pnOldChild);
	virtual bool VirtualInsertChildNode(XMLDOM_Node *pnNewChild, XMLDOM_Node *pnOldChild);

	virtual void VirtualSelectElementsByPattern(XMLDOM_Element *pePatternNode, XMLDOM_NodeList *&pnlOutResultNodes, bool bRecursive=false);
	virtual void VirtualSelectNodes(const string &sPatternString, XMLDOM_NodeList *&pnlOutResultList, bool bCaseInsensitive=false);
	virtual bool VirtualSelectSingleNode(const string &sPatternString, XMLDOM_Node *&pnOutResultNode, bool bCaseInsensitive=false);

public:
	virtual DOMNodeType GetNodeType() const;
	virtual bool IsElementOrDescendant() const;

	void GetAttributes(XMLDOM_NodeList *&pnlOutAttributes) const;
	void GetChildNodes(XMLDOM_NodeList *&pnlOutChildNodes) const;

protected:
	XMLDOM_InternalNodeStorage *GetUnsafeChildNodes() const { return const_cast<XMLDOM_InternalNodeStorage *>(&m_nsChildNodes); }
	size_type GetAttributeNodeCount() const { return m_nAttributeCount; }
	size_type GetStartingChildNodeIndex() const { return GetAttributeNodeCount(); }

public:
	const string &GetTagName() const { return GetNodeName(); }
	const string &GetNamespace() const { return GetNodeNamespace(); }
	bool GetNamespaceByURI(const string& sURI, string& sNamespace) const;
	
	void GetNodeValue(string &sValue) const;
	void SetNodeValue(const string &sValue);

private:
	XMLDOM_Text *GetUnsafeFirstTextChildNode(size_type nStartingIndex, size_type *pnOutChildIndex/*=NULL*/) const { return static_cast<XMLDOM_Text *>(GetUnsafeFirstChildNodeOfType(NODE_TEXT, nStartingIndex, pnOutChildIndex)); }
	void RemoveTextChildNodes(size_type nStartingIndex) { RemoveChildNodesOfType(NODE_TEXT, nStartingIndex); }

protected:
	XMLDOM_Node *GetUnsafeFirstChildNodeOfType(DOMNodeType ntNodeTypeOfInterest, size_type nStartingIndex, size_type *pnOutChildIndex/*=NULL*/) const;
	void RemoveChildNodesOfType(DOMNodeType ntNodeTypeOfInterest, size_type nStartingIndex);
	
public:
	size_type GetAttributeCount() const { return GetAttributeNodeCount(); }
	bool GetAttributeExists(const string &sNameOfInterest) const;
	bool GetAttribute(const string &sNameOfInterest, string &sOutAttributeValue) const;
    void SetAttribute(const string &sNameOfInterest, const string &sAttributeValue, XMLDOM_Attr **ppaOutAttribute=NULL);
    bool RemoveAttribute(const string &sNameOfInterest, XMLDOM_Attr **ppaOutAttribute=NULL);

	bool GetAttributeNode(const string &sNameOfInterest, XMLDOM_Attr *&paOutAttribute) const;
	void SetAttributeNode(XMLDOM_Attr *paInputAttribute, XMLDOM_Attr *&paOutAttribute);
	bool RemoveAttributeNode(XMLDOM_Attr *paAttributeOfInterest, XMLDOM_Attr *&paOutAttribute);
	
private:
	XMLDOM_Attr *GetUnsafeAttributeNode(const string &sNameOfInterest, size_type *pnOutNodeIndex/*=NULL*/) const;
	void AddUnsafeAttributeNode(XMLDOM_Attr *paAttribute);
	void RemoveUnsafeAttributeNode(size_type nAttributeIndex);

	void IncrementAttributeNodeCount() { ++m_nAttributeCount; assert(m_nAttributeCount != 0); }
	void DecrementAttributeNodeCount() { --m_nAttributeCount; assert(m_nAttributeCount != (size_type)0 - (size_type)1); }

public:	
    bool HasChildNodes() const;

	size_type GetChildNodeIndex(XMLDOM_Node *pnChildNode) const;
	bool GetChildNodePreviousSibling(XMLDOM_Node *pnChildNode, XMLDOM_Node *&pnOutPreviousNode) const;
	bool GetChildNodeNextSibling(XMLDOM_Node *pnChildNode, XMLDOM_Node *&pnOutNextNode) const;

	bool GetFirstChildNode(XMLDOM_Node *&pnOutFirstChild) const;
	bool GetLastChildNode(XMLDOM_Node *&pnOutLastChild) const;
	
	void AppendChildNode(XMLDOM_Node *pnNewChild);
	bool RemoveChildNode(XMLDOM_Node *pnOldChild);
	bool ReplaceChildNode(XMLDOM_Node *pnNewChild, XMLDOM_Node *pnOldChild);
	bool InsertChildNode(XMLDOM_Node *pnNewChild, XMLDOM_Node *pnOldChild);

public:
	void RemoveChildNodes();
	void RemoveAttributesAndChildNodes();

	void ResetAttributeNodeCount() { m_nAttributeCount = 0; }

public:
    void GetElementsByTagName(const string &sTagNameOfInterest, XMLDOM_NodeList *&pnlOutResultNodes) const;
	void SelectElementsByPattern(XMLDOM_Element *pePatternNode, XMLDOM_NodeList *&pnlOutResultNodes, bool bRecursive=false) const;

private:
	void SelectElementsByPatternIntoList(XMLDOM_Element *pePatternNode, XMLDOM_NodeList *pnlResultNodes, bool bRecursive) const;
	static bool GetIfElementsMatch(XMLDOM_Element *pePatternNode, XMLDOM_Element *peTestElement);

public:
	void SelectNodes(const string &sPatternString, XMLDOM_NodeList *&pnlOutResultList, bool bCaseInsensitive = false);
	bool SelectSingleNode(const string &sPatternString, XMLDOM_Node *&pnOutResultNode, bool bCaseInsensitive = false);
	
public:
	virtual void GetTitle(string &sNodeTitle);
	virtual void GetText(string &sNodeText);
	virtual void PutText(const string &sNodeText); 
	virtual void Clear();
	
protected:
	virtual string::size_type InternalGetXML(string &sBuffer, EncodingCharset nCharset) const;

private:
	XMLDOM_InternalNodeStorage	m_nsChildNodes; // First m_nAttributeCount of nodes are attributes, the rest are child nodes
	size_type					m_nAttributeCount;
};


typedef XMLDOM_Element XMLDOM_Document_Parent;
class XMLDOM_Document:
public XMLDOM_Document_Parent
{
public:
	XMLDOM_Document();
	
protected:
	XMLDOM_Document(const XMLDOM_Document &dOther, bool bRecursive=false);
	virtual ~XMLDOM_Document(); // Use Release()

public:
	virtual void CloneNode(XMLDOM_Node *&pnNewNode, bool bRecursive) const;

public:
    virtual DOMNodeType GetNodeType() const;
	
public:
	void GetDocumentElement(XMLDOM_Element *&peOutDocumentElement) const;
	
protected:
	XMLDOM_Element *GetUnsafeDocumentElement() const;

private:
	XMLDOM_Element *GetUnsafeFirstElementChildNode(size_type nStartingChildIndex, size_type *pnNodeIndex) const { return static_cast<XMLDOM_Element *>(GetUnsafeFirstChildNodeOfType(NODE_ELEMENT, nStartingChildIndex, pnNodeIndex)); }
	
public:
	//void CreateElement(const string &sTagName, XMLDOM_Element *&peElement);
	void CreateElement(const string &sTagName, const string &sTagNamespace, XMLDOM_Element *&peElement);
	void CreateTextNode(const string &sContent, XMLDOM_Text *&ptText);
	void CreateComment(const string &sContent, XMLDOM_Comment *&pcComment);
	void CreateCDataSection(const string &sData, XMLDOM_CDATASection *&pdCData);
	void CreateAttribute(const string &sAttributeName, XMLDOM_Attr *&paAttribute);
	
public:
	bool LoadXML(const string &sXMLString, XMLDOM_FilterProc fpFilter, void *pvContext);
	bool LoadXMLEx(const string &sXMLString, string &sOutputMessage);
	bool StringToNode(const string &sXMLString, XMLDOM_Node *&pnNewNode);

	EncodingCharset GetCharset() const;
	void SetCharset(EncodingCharset nCharset);
	
protected:
	virtual string::size_type InternalGetXML(string &sBuffer, EncodingCharset nCharset) const;
private:
	EncodingCharset m_nCharSet;
};

string TrimWhiteSpaces(const string& value);

#endif // __XMLPARSERLITE__
