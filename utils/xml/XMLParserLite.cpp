#include "StdAfx.h"

#include "xpath.h"
#include "elxmlparse.h"
#include "atomic_operations.h"
#include "strutils.h"

#include "XMLParserLite.h"

using namespace StrUtils;

#define DEF_STRING_RESERVE		2048
#define DEF_PREDICT_FREQUENCY	4

#if defined(__MULTITHREAD_SAFE) && !(defined(USE_DELPHI_STRING) && defined(SAFE_DELPHI_STRING))

#define SAFE_STRING_CONSTRUCTOR_INITIALIZER(s) (s).c_str(), (s).size()
#define SAFE_STRING_PARAMETER_INITIALIZER(s) string((s).c_str(), (s).size())

#else // #if !defined(__MULTITHREAD_SAFE) || (defined(USE_DELPHI_STRING) && defined(SAFE_DELPHI_STRING))

#define SAFE_STRING_CONSTRUCTOR_INITIALIZER(s) (s)
#define SAFE_STRING_PARAMETER_INITIALIZER(s) (s)

#endif // #if !defined(__MULTITHREAD_SAFE) || (defined(USE_DELPHI_STRING) && defined(SAFE_DELPHI_STRING))


#define CHECK_BUFFER(sBuffer, nSizeUsed, nExtraSize, nNewSizeUsed, nReserve) \
string::size_type nNewSizeUsed = nSizeUsed + (nExtraSize); \
if (sBuffer.size() < nNewSizeUsed) \
{ \
	sBuffer.resize(nNewSizeUsed + (nReserve)); \
}

#define APPEND_SZ(sBuffer, nSizeUsed, szLine, nLineLength) \
if(nLineLength > 0) \
{ \
	CHECK_BUFFER(sBuffer, nSizeUsed, nLineLength, nNewSizeUsed, DEF_STRING_RESERVE); \
	memcpy(&(*sBuffer.begin()) + nSizeUsed, szLine, (nLineLength)); \
	nSizeUsed = nNewSizeUsed; \
}

#define APPEND_STR(sBuffer, nSizeUsed, sLine, nLineLength) \
if(nLineLength > 0) \
{ \
	CHECK_BUFFER(sBuffer, nSizeUsed, nLineLength, nNewSizeUsed, DEF_STRING_RESERVE); \
	memcpy(&(*sBuffer.begin()) + nSizeUsed, sLine.data(), (nLineLength)); \
	nSizeUsed = nNewSizeUsed; \
}

#if (_MSC_VER >= 1200) && (_MSC_VER < 1300) 
#define PREDICT_BUFFER(sBuffer, nSizeUsed, nSizeUsedAtStart, nItemIndex, nItemCount) \
{ \
	if ((nItemIndex) % DEF_PREDICT_FREQUENCY == 0 && (nItemCount)) \
	{ \
		CHECK_BUFFER(sBuffer, nSizeUsed, (string::size_type)(((__int64)(nSizeUsed - nSizeUsedAtStart)) * ((nItemCount) - (nItemIndex)) / nItemCount), nNewSizeUsed, 0); \
	} \
}
#else // #if (_MSC_VER < 1200) || (_MSC_VER >= 1300) 
#define PREDICT_BUFFER(sBuffer, nSizeUsed, nSizeUsedAtStart, nItemIndex, nItemCount) \
{ \
	if ((nItemIndex) % DEF_PREDICT_FREQUENCY == 0 && (nItemCount)) \
	{ \
		CHECK_BUFFER(sBuffer, nSizeUsed, (string::size_type)(((unsigned long long)(nSizeUsed - nSizeUsedAtStart)) * ((nItemCount) - (nItemIndex)) / nItemCount), nNewSizeUsed, 0); \
	} \
}
#endif // #if (_MSC_VER >= 1200) && (_MSC_VER < 1300) 

const char szXMLPparser_UTF8_BOM[] = "\xEF\xBB\xBF";
const char szXMLPparser_UTF8_Encoding[] = "utf-8";
const char szXMLPparser_UTF16_Encoding[] = "utf-16";
const char szXMLPparser_AttrEncoding[] = "encoding";
const char szXMLPparser_AttrVersion[] = "version";
const char szXMLPparser_DefaultVersion[] = "1.0";

static const string g_csEmptyString;


/************************************************************************/
/* XMLDOM_Base                                                          */
/************************************************************************/

XMLDOM_Base::XMLDOM_Base():
	m_ulRefCount(1U)
{
}

XMLDOM_Base::~XMLDOM_Base()
{
	assert(m_ulRefCount == 0U);
}


void XMLDOM_Base::AddRef()
{
	InternalAddRef();
}

void XMLDOM_Base::Release()
{
	InternalRelease();
}


void XMLDOM_Base::InternalAddRef()
{
	Atomic::Increment32((unsigned int *)&m_ulRefCount);
}

void XMLDOM_Base::InternalRelease()
{
	unsigned int ulNewRefCount = Atomic::Decrement32((unsigned int *)&m_ulRefCount) - 1;
	assert(ulNewRefCount != ~0);

	if (ulNewRefCount == 0)
	{
		delete this;
	}
}

/************************************************************************/
/* XMLDOM_InternalNodeStorage                                           */
/************************************************************************/

XMLDOM_InternalNodeStorage::XMLDOM_InternalNodeStorage(const XMLDOM_InternalNodeStorage &nsOther, size_type nBeginIndex, size_type nEndIndex):
	XMLDOM_InternalNodeStorage_Parent()
{
	CloneContents(nsOther, nBeginIndex, nEndIndex);
}

XMLDOM_InternalNodeStorage::~XMLDOM_InternalNodeStorage()
{
	ClearContents();
}

void XMLDOM_InternalNodeStorage::CloneContents(const XMLDOM_InternalNodeStorage &nsOther, size_type nBeginIndex, size_type nEndIndex)
{
	if (nEndIndex == npos)
	{
		nEndIndex = nsOther.GetLength();
	}

	XMLDOM_InternalNodeStorage_Parent::resize(nEndIndex - nBeginIndex);
	
	for (size_type nOtherItemIndex = nBeginIndex; nOtherItemIndex != nEndIndex; ++nOtherItemIndex)
	{
		XMLDOM_Node *pnOtherItem = nsOther.GetUnsafeItem(nOtherItemIndex);
		
		XMLDOM_Node *pnOtherItemClone;
		pnOtherItem->CloneNode(pnOtherItemClone, true);
		
		pnOtherItemClone->InternalAddRef();
		pnOtherItemClone->Release();

		XMLDOM_InternalNodeStorage_Parent::operator[](nOtherItemIndex - nBeginIndex) = pnOtherItemClone;
	}
}

void XMLDOM_InternalNodeStorage::ClearContents()
{
	size_type nItemCount = XMLDOM_InternalNodeStorage_Parent::size();
	for (size_type nItemIndex = 0; nItemIndex != nItemCount; ++nItemIndex)
	{
		XMLDOM_Node *pnItem = XMLDOM_InternalNodeStorage_Parent::operator[](nItemIndex);
	
		pnItem->InternalRelease();
	}
	
	XMLDOM_InternalNodeStorage_Parent::clear();
}


void XMLDOM_InternalNodeStorage::SetItemsParent(XMLDOM_Element *peParentNode, size_type nBeginIndex, size_type nEndIndex)
{
	if (nEndIndex == npos)
	{
		nEndIndex = XMLDOM_InternalNodeStorage_Parent::size();
	}

	for (size_type nItemIndex = nBeginIndex; nItemIndex != nEndIndex; ++nItemIndex)
	{
		XMLDOM_Node *pnItem = XMLDOM_InternalNodeStorage_Parent::operator[](nItemIndex);
		
		pnItem->SetParentNode(peParentNode);
	}
}

void XMLDOM_InternalNodeStorage::SetItemsOwnerDocument(XMLDOM_Document *pdNewDocument)
{
	size_type nItemCount = XMLDOM_InternalNodeStorage_Parent::size();
	for (size_type nItemIndex = 0; nItemIndex != nItemCount; ++nItemIndex)
	{
		XMLDOM_Node *pnItem = XMLDOM_InternalNodeStorage_Parent::operator[](nItemIndex);
		
		pnItem->SetOwnerDocument(pdNewDocument);
	}
}


XMLDOM_Node *XMLDOM_InternalNodeStorage::GetUnsafeItem(size_type nItemIndex) const
{
	assert(nItemIndex < XMLDOM_InternalNodeStorage_Parent::size());
	
	return XMLDOM_InternalNodeStorage_Parent::operator[](nItemIndex);
}


void XMLDOM_InternalNodeStorage::AddItem(XMLDOM_Node *pnNewItem)
{
	XMLDOM_InternalNodeStorage_Parent::push_back(pnNewItem);

	pnNewItem->InternalAddRef();
}

bool XMLDOM_InternalNodeStorage::RemoveItem(size_type nItemIndex)
{
	bool bResult = false;
	
	size_type nItemCount = XMLDOM_InternalNodeStorage_Parent::size();
	if (nItemIndex < nItemCount)
	{
		XMLDOM_InternalNodeStorage_Parent::iterator itItem = XMLDOM_InternalNodeStorage_Parent::begin() + nItemIndex;

		XMLDOM_Node *pnOldItem = *itItem;
		pnOldItem->InternalRelease();
		
		XMLDOM_InternalNodeStorage_Parent::erase(itItem);
		
		bResult = true;
	}
	
	return bResult;
}

bool XMLDOM_InternalNodeStorage::ReplaceItem(size_type nItemIndex, XMLDOM_Node *pnNewItem)
{
	bool bResult = false;
	
	size_type nItemCount = XMLDOM_InternalNodeStorage_Parent::size();
	if (nItemIndex < nItemCount)
	{
		XMLDOM_InternalNodeStorage_Parent::iterator itItem = XMLDOM_InternalNodeStorage_Parent::begin() + nItemIndex;
		
		XMLDOM_Node *pnOldItem = *itItem;
		pnOldItem->InternalRelease();
		
		*itItem = pnNewItem;

		pnNewItem->InternalAddRef();
		
		bResult = true;
	}
	
	return bResult;
}

bool XMLDOM_InternalNodeStorage::InsertItem(size_type nItemIndex, XMLDOM_Node *pnNewItem)
{
	bool bResult = false;
	
	size_type nItemCount = XMLDOM_InternalNodeStorage_Parent::size();
	if (nItemIndex <= nItemCount)
	{
		XMLDOM_InternalNodeStorage_Parent::iterator itItem = XMLDOM_InternalNodeStorage_Parent::begin() + nItemIndex;
		XMLDOM_InternalNodeStorage_Parent::insert(itItem, pnNewItem);
		
		pnNewItem->InternalAddRef();
		
		bResult = true;
	}
	
	return bResult;
}


void XMLDOM_InternalNodeStorage::RemoveItems(size_type nBeginIndex, size_type nEndIndex)
{
	if (nEndIndex == npos)
	{
		nEndIndex = XMLDOM_InternalNodeStorage_Parent::size();
	}

	XMLDOM_InternalNodeStorage_Parent::iterator itBeginItem = XMLDOM_InternalNodeStorage_Parent::begin() + nBeginIndex;
	XMLDOM_InternalNodeStorage_Parent::iterator itEndItem = XMLDOM_InternalNodeStorage_Parent::begin() + nEndIndex;

	for (XMLDOM_InternalNodeStorage_Parent::iterator itItem = itBeginItem; itItem != itEndItem; ++itItem)
	{
		XMLDOM_Node *pnItem = *itItem;

		pnItem->InternalRelease();
	}
	
	XMLDOM_InternalNodeStorage_Parent::erase(itBeginItem, itEndItem);
}


XMLDOM_InternalNodeStorage::size_type XMLDOM_InternalNodeStorage::GetItemIndex(XMLDOM_Node *pnItemOfInterest, size_type nBeginIndex, size_type nEndIndex) const
{
	size_type nResult = npos;

	if (nEndIndex == npos)
	{
		nEndIndex = XMLDOM_InternalNodeStorage_Parent::size();
	}

	for (size_type nItemIndex = nBeginIndex; nItemIndex != nEndIndex; ++nItemIndex)
	{
		XMLDOM_Node *pnItem = XMLDOM_InternalNodeStorage_Parent::operator[](nItemIndex);

		if (pnItem == pnItemOfInterest)
		{
			nResult = nItemIndex;
			break;
		}
	}

	return nResult;
}


/************************************************************************/
/* XMLDOM_NodeList                                                      */
/************************************************************************/

XMLDOM_NodeList::XMLDOM_NodeList(const XMLDOM_NodeList &nlOther):
	XMLDOM_NodeList_Parent(nlOther),
	XMLDOM_NodeList_VectorParent()
{
	AddRefList();
}


XMLDOM_NodeList::~XMLDOM_NodeList()
{
	ReleaseList();
}


void XMLDOM_NodeList::AddRefList()
{
	size_type nItemCount = GetLength();
	for (size_type nItemIndex = 0; nItemIndex != nItemCount; ++nItemIndex)
	{
		XMLDOM_Node *pnItem = GetUnsafeItem(nItemIndex);

		pnItem->AddRef();
	}
}

void XMLDOM_NodeList::ReleaseList()
{
	size_type nItemCount = GetLength();
	for (size_type nItemIndex = 0; nItemIndex != nItemCount; ++nItemIndex)
	{
		XMLDOM_Node *pnItem = GetUnsafeItem(nItemIndex);
		
		pnItem->Release();
	}
}


bool XMLDOM_NodeList::GetItem(size_type nItemIndex, XMLDOM_Node *&pnOutItem) const
{
	bool bResult = false;
	pnOutItem = NULL;
	
	size_type nItemCount = GetLength();
	if (nItemIndex < nItemCount)
	{
		XMLDOM_Node *pnItem = GetUnsafeItem(nItemIndex);
		pnItem->AddRef(); // Addref because item is returned for output
		
		pnOutItem = pnItem;
		bResult = true;
	}
	
	return bResult;
}


void XMLDOM_NodeList::AddItem(XMLDOM_Node *pnItem)
{
	XMLDOM_NodeList_VectorParent::AddItem(pnItem);

	pnItem->AddRef();
}

bool XMLDOM_NodeList::RemoveItem(size_type nItemIndex)
{
	bool bResult = false;

	size_type nItemCount = GetLength();
	if (nItemIndex < nItemCount)
	{
		XMLDOM_Node *pnItem = GetUnsafeItem(nItemIndex);
		pnItem->Release();

		bResult = XMLDOM_NodeList_VectorParent::RemoveItem(nItemIndex);
	}

	return bResult;
}


void XMLDOM_NodeList::FillFromStorage(XMLDOM_InternalNodeStorage *pnsOtherNodes, size_type nBeginIndex, size_type nEndIndex)
{
	assert(GetLength() == 0);

	if (nEndIndex == npos)
	{
		nEndIndex = pnsOtherNodes->GetLength();
	}
	
	for (size_type nNodeIndex = nBeginIndex; nNodeIndex != nEndIndex; ++nNodeIndex)
	{
		XMLDOM_Node *pnOtherNode = pnsOtherNodes->GetUnsafeItem(nNodeIndex);

		AddItem(pnOtherNode);
	}
}


/************************************************************************/
/* XMLDOM_Node                                                          */
/************************************************************************/

XMLDOM_Node::XMLDOM_Node(XMLDOM_Document *pdOwnerDocument):
	XMLDOM_Node_Parent(),
	m_ulExternalRefCount(1UL),
	m_pdOwnerDocument(pdOwnerDocument),
	m_peParentNode(NULL),
	m_sNodeName(),
	m_sNodeNamespace()
{
	static_cast<XMLDOM_Node *>(pdOwnerDocument)->InternalAddRef();
}

XMLDOM_Node::XMLDOM_Node(const XMLDOM_Node &nOther):
	XMLDOM_Node_Parent(),
	m_ulExternalRefCount(1UL),
	m_pdOwnerDocument(nOther.m_pdOwnerDocument),
	m_peParentNode(NULL),
	m_sNodeName(SAFE_STRING_CONSTRUCTOR_INITIALIZER(nOther.m_sNodeName)),
	m_sNodeNamespace(SAFE_STRING_CONSTRUCTOR_INITIALIZER(nOther.m_sNodeNamespace))
{
	static_cast<XMLDOM_Node *>(m_pdOwnerDocument)->InternalAddRef();
}


XMLDOM_Node::~XMLDOM_Node()
{
	assert(GetUnsafeParentNode() == NULL);
}


void XMLDOM_Node::AddRef()
{
	XMLDOM_Node_Parent::AddRef();

	unsigned int ulExternalRefCount = Atomic::Increment32((unsigned int *)&m_ulExternalRefCount) + 1U;
	assert(ulExternalRefCount != 0);

	if (ulExternalRefCount == 1U)
	{
		static_cast<XMLDOM_Node *>(m_pdOwnerDocument)->InternalAddRef();
	}
}

void XMLDOM_Node::Release()
{
	unsigned int ulExternalRefCount = Atomic::Decrement32((unsigned int *)&m_ulExternalRefCount) - 1U;
	assert(ulExternalRefCount != 0U - 1U);

	if (ulExternalRefCount == 0U)
	{
		static_cast<XMLDOM_Node *>(m_pdOwnerDocument)->InternalRelease();
	}

	XMLDOM_Node_Parent::Release();
}

void XMLDOM_Node::SetOwnerDocument(XMLDOM_Document *pdNewDocument)
{
	assert(!m_peParentNode || static_cast<XMLDOM_Node *>(m_peParentNode)->GetUnsafeOwnerDocument() == pdNewDocument); // Owner can be changed only for nodes which are not inserted into tree yet

	XMLDOM_Document *pdOldDocument = GetUnsafeOwnerDocument();
	
	if (pdNewDocument != pdOldDocument)
	{
		m_pdOwnerDocument = pdNewDocument;

		// If current node is not inserted into tree yet it should not be accessed from
		// multiple threads and therefore it should be safe to access its reference count
		unsigned int ulExternalRefCount = GetExternalRefCount();
		if (ulExternalRefCount)
		{
			static_cast<XMLDOM_Node *>(pdOldDocument)->InternalRelease();
			static_cast<XMLDOM_Node *>(pdNewDocument)->InternalAddRef();
		}

		assert(ulExternalRefCount == GetExternalRefCount()); // Assert that reference count did not change
	}
}


const string &XMLDOM_Node::VirtualGetNodeName() const
{
	return g_csEmptyString;
}

void XMLDOM_Node::VirtualGetNodeValue(string &sValue) const
{
	sValue.erase();
}

void XMLDOM_Node::VirtualSetNodeName(const string &sNodeName)
{
	assert(false);
}

void XMLDOM_Node::VirtualSetNodeValue(const string &sNodeValue)
{
	assert(false);
}


bool XMLDOM_Node::VirtualHasChildNodes() const
{
	return false;
}

void XMLDOM_Node::VirtualGetChildNodes(XMLDOM_NodeList *&pnlChildNodes)
{
	pnlChildNodes = new XMLDOM_NodeList();
}

void XMLDOM_Node::VirtualRemoveChildNodes()
{
	// Do nothing
}


bool XMLDOM_Node::VirtualGetFirstChildNode(XMLDOM_Node *&pnChildNode)
{
	pnChildNode = NULL;
	return false;
}

bool XMLDOM_Node::VirtualGetLastChildNode(XMLDOM_Node *&pnChildNode)
{
	pnChildNode = NULL;
	return false;
}

bool XMLDOM_Node::VirtualGetPreviousSiblingNode(XMLDOM_Node *&pnChildNode)
{
	return GetPreviousSiblingNode(pnChildNode);
}

bool XMLDOM_Node::VirtualGetNextSiblingNode(XMLDOM_Node *&pnChildNode)
{
	return GetNextSiblingNode(pnChildNode);
}

XMLDOM_Node::size_type XMLDOM_Node::VirtualGetIndexInParent() const
{
	return GetIndexInParent();
}


void XMLDOM_Node::VirtualAppendChildNode(XMLDOM_Node *pnNewChild)
{
	assert(false);
}

bool XMLDOM_Node::VirtualRemoveChildNode(XMLDOM_Node *pnOldChild)
{
	assert(false);

	return false;
}

bool XMLDOM_Node::VirtualReplaceChildNode(XMLDOM_Node *pnNewChild, XMLDOM_Node *pnOldChild)
{
	assert(false);

	return false;
}

bool XMLDOM_Node::VirtualInsertChildNode(XMLDOM_Node *pnNewChild, XMLDOM_Node *pnOldChild)
{
	assert(false);

	return false;
}


void XMLDOM_Node::VirtualSelectElementsByPattern(XMLDOM_Element *pePatternNode, XMLDOM_NodeList *&pnlOutResultNodes, bool bRecursive)
{
	pnlOutResultNodes = new XMLDOM_NodeList();
}

void XMLDOM_Node::VirtualSelectNodes(const string &sPatternString, XMLDOM_NodeList *&pnlOutResultList, bool bCaseInsensitive)
{
	pnlOutResultList = new XMLDOM_NodeList();
}

bool XMLDOM_Node::VirtualSelectSingleNode(const string &sPatternString, XMLDOM_Node *&pnOutResultNode, bool bCaseInsensitive)
{
	pnOutResultNode = NULL;
	return false;
}


void XMLDOM_Node::GetOwnerDocument(XMLDOM_Document *&pdOutOwnerDocument) const
{
	XMLDOM_Document *pdOwnerDocument = GetUnsafeOwnerDocument();
	
	pdOwnerDocument->AddRef();

	pdOutOwnerDocument = pdOwnerDocument;
}

bool XMLDOM_Node::GetParentNode(XMLDOM_Element *&peOutParentNode) const
{
	bool bResult = false;
		
	XMLDOM_Element *peParentNode = GetUnsafeParentNode();

	if (peParentNode)
	{
		peParentNode->AddRef();

		bResult = true;
	}

	peOutParentNode = peParentNode;
	return bResult;
}


bool XMLDOM_Node::IsElementOrDescendant() const
{
	return false;
}


XMLDOM_Node::size_type XMLDOM_Node::GetIndexInParent() const
{
	size_type nResult = npos;

	XMLDOM_Element *peParentNode = GetUnsafeParentNode();
	
	if (peParentNode)
	{
		nResult = peParentNode->GetChildNodeIndex(const_cast<XMLDOM_Node *>(this));
	}

	return nResult;
}

void XMLDOM_Node::RemoveFromParent()
{
	XMLDOM_Element *peParentNode = GetUnsafeParentNode();
	
	if (peParentNode)
	{
		peParentNode->RemoveChildNode(this);
	}
}


bool XMLDOM_Node::GetPreviousSiblingNode(XMLDOM_Node *&pnOutPreviousNode) const
{
	bool bResult = false;
	pnOutPreviousNode = NULL;

	XMLDOM_Element *peParentNode = GetUnsafeParentNode();
	
	if (peParentNode)
	{
		bResult = peParentNode->GetChildNodePreviousSibling(const_cast<XMLDOM_Node *>(this), pnOutPreviousNode);
	}

	return bResult;
}

bool XMLDOM_Node::GetNextSiblingNode(XMLDOM_Node *&pnOutNextNode) const
{
	bool bResult = false;
	pnOutNextNode = NULL;
	
	XMLDOM_Element *peParentNode = GetUnsafeParentNode();
	
	if (peParentNode)
	{
		bResult = peParentNode->GetChildNodeNextSibling(const_cast<XMLDOM_Node *>(this), pnOutNextNode);
	}
	
	return bResult;
}


void XMLDOM_Node::GetXML(string &sXMLString, EncodingCharset nCharset)
{
	string::size_type nSizeUsed = InternalGetXML(sXMLString, nCharset);
	sXMLString.resize(nSizeUsed);
}


/************************************************************************/
/* XMLDOM_CharacterData                                                 */
/************************************************************************/

XMLDOM_CharacterData::XMLDOM_CharacterData(XMLDOM_Document *pdOwnerDocument, const string &sData):
	XMLDOM_CharacterData_Parent(pdOwnerDocument)
{
	SetData(sData);
}

XMLDOM_CharacterData::XMLDOM_CharacterData(const XMLDOM_CharacterData &cdOther):
	XMLDOM_CharacterData_Parent(cdOther)
{
	const string &sOtherData = cdOther.GetData();
	SetData(SAFE_STRING_PARAMETER_INITIALIZER(sOtherData));
}

XMLDOM_CharacterData::~XMLDOM_CharacterData()
{
	// Do nothing
}


void XMLDOM_CharacterData::VirtualGetNodeValue(string &sValue) const
{
	sValue = GetData();
}

void XMLDOM_CharacterData::VirtualSetNodeValue(const string &sNodeValue)
{
	SetData(sNodeValue);
}


void XMLDOM_CharacterData::AppendData(const string &sNewData)
{
	string sData = GetData();

	sData.append(sNewData.c_str());
	
	SetData(sData);
}

void XMLDOM_CharacterData::InsertData(string::size_type nOffset, const  string &sNewData)
{
	string sData = GetData();
	
	sData.insert(nOffset, sNewData);
	
	SetData(sData);
}

void XMLDOM_CharacterData::DeleteData(string::size_type nOffset, string::size_type nCount)
{
	string sData = GetData();
	
	sData.erase(nOffset, nCount);
	
	SetData(sData);
}

void XMLDOM_CharacterData::ReplaceData(string::size_type nOffset, string::size_type nCount, const string &sNewData)
{
	string sData = GetData();

	sData.replace(nOffset, nCount, sNewData);
	
	SetData(sData);
}


void XMLDOM_CharacterData::GetTitle(string &sNodeTitle)
{
	sNodeTitle.erase();
}

void XMLDOM_CharacterData::GetText(string &sNodeText)
{
	sNodeText = GetData();
}

void XMLDOM_CharacterData::PutText(const string &sNodeText)
{
	SetData(sNodeText);
}

void XMLDOM_CharacterData::Clear()
{
	SetData(g_csEmptyString);
}


/************************************************************************/
/* XMLDOM_Attr                                                          */
/************************************************************************/

XMLDOM_Attr::XMLDOM_Attr(XMLDOM_Document *pdOwnerDocument, const string &sName):
	XMLDOM_Attr_Parent(pdOwnerDocument),
	m_sValue()
{
	SetNodeName(sName);
}

XMLDOM_Attr::XMLDOM_Attr(const XMLDOM_Attr &aOther):
	XMLDOM_Attr_Parent(aOther),
	m_sValue(SAFE_STRING_CONSTRUCTOR_INITIALIZER(aOther.m_sValue))
{
}

XMLDOM_Attr::~XMLDOM_Attr()
{
	// Do nothing
}


void XMLDOM_Attr::CloneNode(XMLDOM_Node *&pnNewNode, bool bRecursive) const
{
	pnNewNode = new XMLDOM_Attr(*this);
}


const string &XMLDOM_Attr::VirtualGetNodeName() const
{
	return GetName();
}

void XMLDOM_Attr::VirtualGetNodeValue(string &sValue) const
{
	sValue = GetValue();
}

void XMLDOM_Attr::VirtualSetNodeName(const string &sNodeName)
{
	SetName(sNodeName);
}

void XMLDOM_Attr::VirtualSetNodeValue(const string &sNodeValue)
{
	SetValue(sNodeValue);
}


bool XMLDOM_Attr::VirtualGetPreviousSiblingNode(XMLDOM_Node *&pnChildNode)
{
	assert(false);

	pnChildNode = NULL;
	return false;
}

bool XMLDOM_Attr::VirtualGetNextSiblingNode(XMLDOM_Node *&pnChildNode)
{
	assert(false);

	pnChildNode = NULL;
	return false;
}

XMLDOM_Node::size_type XMLDOM_Attr::VirtualGetIndexInParent() const
{
	assert(false);

	return npos;
}


DOMNodeType XMLDOM_Attr::GetNodeType() const
{
	return NODE_ATTRIBUTE;
}


void XMLDOM_Attr::GetTitle(string &sNodeTitle)
{
	sNodeTitle = GetName();
}

void XMLDOM_Attr::GetText(string &sNodeText)
{
	sNodeText = GetValue();
}

void XMLDOM_Attr::PutText(const string &sNodeText)
{
	SetValue(sNodeText);
}

void XMLDOM_Attr::Clear()
{
	SetValue(g_csEmptyString);
}


string::size_type XMLDOM_Attr::InternalGetXML(string &sBuffer, EncodingCharset nCharset) const
{
	string::size_type nSizeUsed = 0;
	
	APPEND_SZ(sBuffer, nSizeUsed, NODE_ATTR_STAG, STATIC_LENGTH(NODE_ATTR_STAG));
	
	const string &sName = GetName();
	string::size_type nNameSize = sName.size();
	APPEND_STR(sBuffer, nSizeUsed, sName, nNameSize);
	
	APPEND_SZ(sBuffer, nSizeUsed, NODE_ATTR_MTAG, STATIC_LENGTH(NODE_ATTR_MTAG));
	
	const string &sValue = GetValue();
	string sConvertedFNodeValue;
	if(nCharset == ECS_NONE)
	{
		nCharset = GetUnsafeOwnerDocument()->GetCharset();
	}
	ConvertStringToSpec(sValue, sConvertedFNodeValue, nCharset);
	
	string::size_type nConvertedSize = sConvertedFNodeValue.size();
	APPEND_STR(sBuffer, nSizeUsed, sConvertedFNodeValue, nConvertedSize);
	
	APPEND_SZ(sBuffer, nSizeUsed, NODE_ATTR_ETAG, STATIC_LENGTH(NODE_ATTR_ETAG));
	
	return nSizeUsed;
}


/************************************************************************/
/* XMLDOM_Text                                                          */
/************************************************************************/

XMLDOM_Text::XMLDOM_Text(XMLDOM_Document *pdOwnerDocument, const string &sContent):
	XMLDOM_Text_Parent(pdOwnerDocument, sContent)
{
}

XMLDOM_Text::XMLDOM_Text(const XMLDOM_Text &tOther):
	XMLDOM_Text_Parent(tOther)
{
}


XMLDOM_Text::~XMLDOM_Text()
{
	// Do nothing
}


void XMLDOM_Text::CloneNode(XMLDOM_Node *&pnNewNode, bool bRecursive) const
{
    pnNewNode = new XMLDOM_Text(*this);
}

	
DOMNodeType XMLDOM_Text::GetNodeType() const
{
	return NODE_TEXT;
}


string::size_type XMLDOM_Text::InternalGetXML(string &sBuffer, EncodingCharset nCharset) const
{
	const string &sContent = GetData();
	string sConvertedFNodeValue;
	if(nCharset == ECS_NONE)
	{
		nCharset = GetUnsafeOwnerDocument()->GetCharset();
	}
	ConvertStringToSpec(sContent, sConvertedFNodeValue, nCharset);
	
	string::size_type nSizeUsed = 0;
	
	string::size_type nConvertedSize = sConvertedFNodeValue.size();

	APPEND_STR(sBuffer, nSizeUsed, sConvertedFNodeValue, nConvertedSize);

	return nSizeUsed;
}


/************************************************************************/
/* XMLDOM_Comment                                                       */
/************************************************************************/

XMLDOM_Comment::XMLDOM_Comment(XMLDOM_Document *pdOwnerDocument, const string &sContent):
	XMLDOM_Comment_Parent(pdOwnerDocument, sContent)
{
}

XMLDOM_Comment::XMLDOM_Comment(const XMLDOM_Comment &cOther):
	XMLDOM_Comment_Parent(cOther)
{
}


XMLDOM_Comment::~XMLDOM_Comment()
{
	// Do nothing
}

	
DOMNodeType XMLDOM_Comment::GetNodeType() const
{
	return NODE_COMMENT;
}


void XMLDOM_Comment::CloneNode(XMLDOM_Node *&pnNewNode, bool bRecursive) const
{
	pnNewNode = new XMLDOM_Comment(*this);
}

	
string::size_type XMLDOM_Comment::InternalGetXML(string &sBuffer, EncodingCharset nCharset) const
{
	string::size_type nSizeUsed = 0;
	
	APPEND_SZ(sBuffer, nSizeUsed, NODE_COMMENT_STAG, STATIC_LENGTH(NODE_COMMENT_STAG));
	
	const string &sContent = GetData();
	string::size_type nValueSize = sContent.size();
	APPEND_STR(sBuffer, nSizeUsed, sContent, nValueSize);
	
	APPEND_SZ(sBuffer, nSizeUsed, NODE_COMMENT_ETAG, STATIC_LENGTH(NODE_COMMENT_ETAG));
	
	return nSizeUsed;
}


/************************************************************************/
/* XMLDOM_CDATASection                                                  */
/************************************************************************/

XMLDOM_CDATASection::XMLDOM_CDATASection(XMLDOM_Document *pdOwnerDocument, const string &sContent):
	XMLDOM_CDATASection_Parent(pdOwnerDocument, sContent)
{
}

XMLDOM_CDATASection::XMLDOM_CDATASection(const XMLDOM_CDATASection &cdOther):
	XMLDOM_CDATASection_Parent(cdOther)
{
}


XMLDOM_CDATASection::~XMLDOM_CDATASection()
{
	// Do nothing
}

	
void XMLDOM_CDATASection::CloneNode(XMLDOM_Node *&pnNewNode, bool bRecursive) const
{
	pnNewNode = new XMLDOM_CDATASection(*this);
}


DOMNodeType XMLDOM_CDATASection::GetNodeType() const
{
	return NODE_CDATA_SECTION;
}


string::size_type XMLDOM_CDATASection::InternalGetXML(string &sBuffer, EncodingCharset nCharset) const
{
	string::size_type nSizeUsed = 0;
	
	APPEND_SZ(sBuffer, nSizeUsed, NODE_CDATA_STAG, STATIC_LENGTH(NODE_CDATA_STAG));
	
	const string &sContent = GetData();
	string::size_type nValueSize = sContent.size();
	APPEND_STR(sBuffer, nSizeUsed, sContent, nValueSize);
	
	APPEND_SZ(sBuffer, nSizeUsed, NODE_CDATA_ETAG, STATIC_LENGTH(NODE_CDATA_ETAG));
	
	return nSizeUsed;
}


/************************************************************************/
/* XMLDOM_Element                                                       */
/************************************************************************/

XMLDOM_Element::XMLDOM_Element(XMLDOM_Document *pdOwnerDocument, const string &sElementName, 
							   const string &sElementNamespace /* = "" */): 
	XMLDOM_Element_Parent(pdOwnerDocument),
	m_nsChildNodes(),
	m_nAttributeCount(0)
{
	SetNodeName(sElementName);
	SetNodeNamespace(sElementNamespace);
	
	assert((unsigned int)XMLDOM_InternalNodeStorage::npos == (unsigned int)npos); // used as assumption through code
}

XMLDOM_Element::XMLDOM_Element(const XMLDOM_Element &eOther, bool bRecursive):
	XMLDOM_Element_Parent(eOther),
	m_nsChildNodes(eOther.m_nsChildNodes, 0, bRecursive ? npos : eOther.m_nAttributeCount),
	m_nAttributeCount(eOther.m_nAttributeCount)
{
	m_nsChildNodes.SetItemsParent(this);

	assert((unsigned int)XMLDOM_InternalNodeStorage::npos == (unsigned int)npos); // used as assumption through code
}


XMLDOM_Element::~XMLDOM_Element()
{
	m_nsChildNodes.SetItemsParent(NULL);
}
	

void XMLDOM_Element::CloneNode(XMLDOM_Node *&pnNewNode, bool bRecursive) const
{
	pnNewNode = new XMLDOM_Element(*this, bRecursive);
}

void XMLDOM_Element::SetOwnerDocument(XMLDOM_Document *pdNewDocument)
{
	XMLDOM_Element_Parent::SetOwnerDocument(pdNewDocument);

	XMLDOM_InternalNodeStorage *pnsChildNodes = GetUnsafeChildNodes();
	pnsChildNodes->SetItemsOwnerDocument(pdNewDocument);
}


const string &XMLDOM_Element::VirtualGetNodeName() const
{
	return GetTagName();
}

void XMLDOM_Element::VirtualGetNodeValue(string &sValue) const
{
	GetNodeValue(sValue);
}

void XMLDOM_Element::VirtualSetNodeName(const string &sNodeName)
{
	SetNodeName(sNodeName);
}

void XMLDOM_Element::VirtualSetNodeValue(const string &sNodeValue)
{
	SetNodeValue(sNodeValue);
}


bool XMLDOM_Element::VirtualHasChildNodes() const
{
	return HasChildNodes();
}

void XMLDOM_Element::VirtualGetChildNodes(XMLDOM_NodeList *&pnlChildNodes)
{
	GetChildNodes(pnlChildNodes);
}

void XMLDOM_Element::VirtualRemoveChildNodes()
{
	RemoveChildNodes();
}


bool XMLDOM_Element::VirtualGetFirstChildNode(XMLDOM_Node *&pnChildNode)
{
	return GetFirstChildNode(pnChildNode);
}

bool XMLDOM_Element::VirtualGetLastChildNode(XMLDOM_Node *&pnChildNode)
{
	return GetLastChildNode(pnChildNode);
}


void XMLDOM_Element::VirtualAppendChildNode(XMLDOM_Node *pnNewChild)
{
	AppendChildNode(pnNewChild);
}

bool XMLDOM_Element::VirtualRemoveChildNode(XMLDOM_Node *pnOldChild)
{
	return RemoveChildNode(pnOldChild);
}

bool XMLDOM_Element::VirtualReplaceChildNode(XMLDOM_Node *pnNewChild, XMLDOM_Node *pnOldChild)
{
	return ReplaceChildNode(pnNewChild, pnOldChild);
}

bool XMLDOM_Element::VirtualInsertChildNode(XMLDOM_Node *pnNewChild, XMLDOM_Node *pnOldChild)
{
	return InsertChildNode(pnNewChild, pnOldChild);
}


void XMLDOM_Element::VirtualSelectElementsByPattern(XMLDOM_Element *pePatternNode, XMLDOM_NodeList *&pnlOutResultNodes, bool bRecursive)
{
	SelectElementsByPattern(pePatternNode, pnlOutResultNodes, bRecursive);
}

void XMLDOM_Element::VirtualSelectNodes(const string &sPatternString, XMLDOM_NodeList *&pnlOutResultList, bool bCaseInsensitive)
{
	SelectNodes(sPatternString, pnlOutResultList, bCaseInsensitive);
}

bool XMLDOM_Element::VirtualSelectSingleNode(const string &sPatternString, XMLDOM_Node *&pnOutResultNode, bool bCaseInsensitive)
{
	return SelectSingleNode(sPatternString, pnOutResultNode, bCaseInsensitive);
}


DOMNodeType XMLDOM_Element::GetNodeType() const
{
	return NODE_ELEMENT;
}

bool XMLDOM_Element::IsElementOrDescendant() const
{
	return true;
}


void XMLDOM_Element::GetAttributes(XMLDOM_NodeList *&pnlOutAttributes) const
{
	XMLDOM_NodeList *pnlAttributeList = new XMLDOM_NodeList();

	if (pnlAttributeList)
	{
		XMLDOM_InternalNodeStorage *pnsChildNodes = GetUnsafeChildNodes();
		size_type nAttributeCount = GetAttributeNodeCount();
		pnlAttributeList->FillFromStorage(pnsChildNodes, 0, nAttributeCount);
	}

	pnlOutAttributes = pnlAttributeList;
}

void XMLDOM_Element::GetChildNodes(XMLDOM_NodeList *&pnlOutChildNodes) const
{
	XMLDOM_NodeList *pnlChildNodeList = new XMLDOM_NodeList();

	if (pnlChildNodeList)
	{
		XMLDOM_InternalNodeStorage *pnsChildNodes = GetUnsafeChildNodes();
		size_type nAttributeCount = GetAttributeNodeCount();
		pnlChildNodeList->FillFromStorage(pnsChildNodes, nAttributeCount, XMLDOM_NodeList::npos);
	}

	pnlOutChildNodes = pnlChildNodeList;
}


void XMLDOM_Element::GetNodeValue(string &sValue) const
{
	sValue.erase();

	size_type nStartingNodeIndex = GetStartingChildNodeIndex();

	while (true)
	{
		size_type nChildIndex;
		XMLDOM_Text *ptTextChild = GetUnsafeFirstTextChildNode(nStartingNodeIndex, &nChildIndex);
		if (!ptTextChild)
		{
			break;
		}

		const string &sNodeData = ptTextChild->GetData();
		sValue.append(sNodeData.c_str());

		nStartingNodeIndex = nChildIndex + 1;
	}
}

void XMLDOM_Element::SetNodeValue(const string &sValue)
{
	size_type nStartingNodeIndex = GetStartingChildNodeIndex();

	size_type nChildIndex;
	XMLDOM_Text *ptTextChild = GetUnsafeFirstTextChildNode(nStartingNodeIndex, &nChildIndex);
	if (ptTextChild)
	{
		ptTextChild->SetData(sValue);

		RemoveTextChildNodes(nChildIndex + 1);
	}
	else
	{
		XMLDOM_Document *pdOwnerDocument = GetUnsafeOwnerDocument();
		ptTextChild = new XMLDOM_Text(pdOwnerDocument, sValue);

		AppendChildNode(ptTextChild);

		ptTextChild->Release();
	}
}


XMLDOM_Node *XMLDOM_Element::GetUnsafeFirstChildNodeOfType(DOMNodeType ntNodeTypeOfInterest, size_type nStartingIndex, size_type *pnOutChildIndex/*=NULL*/) const
{
	XMLDOM_Node *pnResult = NULL;

	XMLDOM_InternalNodeStorage *pnsChildNodes = GetUnsafeChildNodes();
	
	size_type nChildCount = pnsChildNodes->GetLength();
	for (size_type nChildIndex = nStartingIndex; nChildIndex != nChildCount; ++nChildIndex)
	{
		XMLDOM_Node *pnChildNode = pnsChildNodes->GetUnsafeItem(nChildIndex);
		assert(pnChildNode->GetNodeType() != NODE_ATTRIBUTE);
		
		DOMNodeType ntChildType = pnChildNode->GetNodeType();
		if (ntChildType == ntNodeTypeOfInterest)
		{
			if (pnOutChildIndex)
			{
				*pnOutChildIndex = nChildIndex;
			}

			pnResult = pnChildNode;
			break;
		}
	}

	return pnResult;
}

void XMLDOM_Element::RemoveChildNodesOfType(DOMNodeType ntNodeTypeOfInterest, size_type nStartingIndex)
{
	XMLDOM_InternalNodeStorage *pnsChildNodes = GetUnsafeChildNodes();
	
	size_type nChildCount = pnsChildNodes->GetLength();
	for (size_type nChildIndex = nStartingIndex; nChildIndex != nChildCount; )
	{
		XMLDOM_Node *pnChildNode = pnsChildNodes->GetUnsafeItem(nChildIndex);
		assert(pnChildNode->GetNodeType() != NODE_ATTRIBUTE);
		
		DOMNodeType ntChildType = pnChildNode->GetNodeType();
		if (ntChildType == ntNodeTypeOfInterest)
		{
			pnsChildNodes->RemoveItem(nChildIndex);
			--nChildCount;
		}
		else
		{
			++nChildIndex;
		}
	}
}


bool XMLDOM_Element::GetAttributeExists(const string &sNameOfInterest) const
{
	XMLDOM_Attr *paAttribute = GetUnsafeAttributeNode(sNameOfInterest, NULL);
	return paAttribute != NULL;
}

bool XMLDOM_Element::GetAttribute(const string &sNameOfInterest, string &sOutAttributeValue) const
{
	bool bResult = false;
	sOutAttributeValue.erase();

	XMLDOM_Attr *paAttribute = GetUnsafeAttributeNode(sNameOfInterest, NULL);

	if (paAttribute)
	{
		const string &sAttributeValue = paAttribute->GetValue();

		sOutAttributeValue = sAttributeValue;
		bResult = true;
	}

	return bResult;
}

void XMLDOM_Element::SetAttribute(const string &sNameOfInterest, const string &sAttributeValue, XMLDOM_Attr **ppaOutAttribute/*=NULL*/)
{
	XMLDOM_Attr *paAttribute = GetUnsafeAttributeNode(sNameOfInterest, NULL);
	
	if (!paAttribute)
	{
		XMLDOM_Document *pdOwnerDocument = GetUnsafeOwnerDocument();
		paAttribute = new XMLDOM_Attr(pdOwnerDocument, sNameOfInterest);

		paAttribute->SetParentNode(this);

		AddUnsafeAttributeNode(paAttribute);

		paAttribute->Release();
	}

	paAttribute->SetValue(sAttributeValue);

	if (ppaOutAttribute)
	{
		paAttribute->AddRef();
		*ppaOutAttribute = paAttribute;
	}
}

bool XMLDOM_Element::RemoveAttribute(const string &sNameOfInterest, XMLDOM_Attr **ppaOutAttribute/*=NULL*/)
{
	bool bResult = false;
	if (ppaOutAttribute)
	{
		*ppaOutAttribute = NULL;
	}

	size_type nAttributeIndex;
	XMLDOM_Attr *paAttribute = GetUnsafeAttributeNode(sNameOfInterest, &nAttributeIndex);
	
	if (paAttribute)
	{
		if (ppaOutAttribute)
		{
			paAttribute->AddRef();
			*ppaOutAttribute = paAttribute;
		}

		paAttribute->SetParentNode(NULL);

		RemoveUnsafeAttributeNode(nAttributeIndex);

		bResult = true;
	}

	return bResult;
}


bool XMLDOM_Element::GetAttributeNode(const string &sNameOfInterest, XMLDOM_Attr *&paOutAttribute) const
{
	bool bResult = false;
	paOutAttribute = NULL;

	XMLDOM_Attr *paAttribute = GetUnsafeAttributeNode(sNameOfInterest, NULL);

	if (paAttribute)
	{
		paAttribute->AddRef();
		
		paOutAttribute = paAttribute;
		bResult = true;
	}

	return bResult;
}

void XMLDOM_Element::SetAttributeNode(XMLDOM_Attr *paInputAttribute, XMLDOM_Attr *&paOutAttribute)
{
	// It is safer to just assign new attribute from strings rather than 
	// inspecting input attribute parent and owner document
	const string &sNameOfInterest = paInputAttribute->GetName();
	const string &sAttributeValue = paInputAttribute->GetValue();
	SetAttribute(sNameOfInterest, sAttributeValue, &paOutAttribute);
}

bool XMLDOM_Element::RemoveAttributeNode(XMLDOM_Attr *paAttributeOfInterest, XMLDOM_Attr *&paOutAttribute)
{
	const string &sNameOfInterest = paAttributeOfInterest->GetName();
	return RemoveAttribute(sNameOfInterest, &paOutAttribute);
}


XMLDOM_Attr *XMLDOM_Element::GetUnsafeAttributeNode(const string &sNameOfInterest, size_type *pnOutNodeIndex/*=NULL*/) const
{
	XMLDOM_Attr *paResult = NULL;

	XMLDOM_InternalNodeStorage *pnsChildNodes = GetUnsafeChildNodes();
	
	size_type nAttributeCount = GetAttributeNodeCount();
	for (size_type nAttributeIndex = 0; nAttributeIndex != nAttributeCount; ++nAttributeIndex)
	{
		XMLDOM_Attr *paAttribute = static_cast<XMLDOM_Attr *>(pnsChildNodes->GetUnsafeItem(nAttributeIndex));
		assert(paAttribute->GetNodeType() == NODE_ATTRIBUTE);
		
		const string &sAttributeName = paAttribute->GetName();
		if (sAttributeName == sNameOfInterest)
		{
			if (pnOutNodeIndex)
			{
				*pnOutNodeIndex = nAttributeIndex;
			}

			paResult = paAttribute;
			break;
		}
	}
	
	return paResult;
}

void XMLDOM_Element::AddUnsafeAttributeNode(XMLDOM_Attr *paAttribute)
{
	XMLDOM_InternalNodeStorage *pnsChildNodes = GetUnsafeChildNodes();
	size_type nAttributeCount = GetAttributeNodeCount();

	pnsChildNodes->InsertItem(nAttributeCount, paAttribute);

	IncrementAttributeNodeCount();
}

void XMLDOM_Element::RemoveUnsafeAttributeNode(size_type nAttributeIndex)
{
	XMLDOM_InternalNodeStorage *pnsChildNodes = GetUnsafeChildNodes();
	pnsChildNodes->RemoveItem(nAttributeIndex);

	DecrementAttributeNodeCount();
}


bool XMLDOM_Element::HasChildNodes() const
{
	XMLDOM_InternalNodeStorage *pnsChildNodes = GetUnsafeChildNodes();
	
	size_type nChildCount = pnsChildNodes->GetLength();
	size_type nStartingChildIndex = GetStartingChildNodeIndex();
	return nChildCount != nStartingChildIndex;
}


XMLDOM_Element::size_type XMLDOM_Element::GetChildNodeIndex(XMLDOM_Node *pnChildNode) const
{
	XMLDOM_InternalNodeStorage *pnsChildNodes = GetUnsafeChildNodes();
	size_type nStartingChildIndex = GetStartingChildNodeIndex();
	
	size_type nChildIndex = pnsChildNodes->GetItemIndex(pnChildNode, nStartingChildIndex, npos);
	return nChildIndex;
}

bool XMLDOM_Element::GetChildNodePreviousSibling(XMLDOM_Node *pnChildNode, XMLDOM_Node *&pnOutPreviousNode) const
{
	bool bResult = false;
	pnOutPreviousNode = NULL;

	XMLDOM_InternalNodeStorage *pnsChildNodes = GetUnsafeChildNodes();
	size_type nStartingChildIndex = GetStartingChildNodeIndex();
	
	size_type nChildIndex = pnsChildNodes->GetItemIndex(pnChildNode, nStartingChildIndex, npos);
	if (nChildIndex != npos && nChildIndex != nStartingChildIndex)
	{
		XMLDOM_Node *pnPreviousNode = pnsChildNodes->GetUnsafeItem(nChildIndex - 1);
		assert(pnChildNode->GetNodeType() != NODE_ATTRIBUTE);
		
		pnPreviousNode->AddRef();

		pnOutPreviousNode = pnPreviousNode;
		bResult = true;
	}

	return bResult;
}

bool XMLDOM_Element::GetChildNodeNextSibling(XMLDOM_Node *pnChildNode, XMLDOM_Node *&pnOutNextNode) const
{
	bool bResult = false;
	pnOutNextNode = NULL;
	
	XMLDOM_InternalNodeStorage *pnsChildNodes = GetUnsafeChildNodes();
	size_type nStartingChildIndex = GetStartingChildNodeIndex();
	
	size_type nChildIndex = pnsChildNodes->GetItemIndex(pnChildNode, nStartingChildIndex, npos);
	if (nChildIndex != npos)
	{
		size_type nChildCount = pnsChildNodes->GetLength();
		if (nChildIndex != nChildCount - 1)
		{
			XMLDOM_Node *pnNextNode = pnsChildNodes->GetUnsafeItem(nChildIndex + 1);
			assert(pnChildNode->GetNodeType() != NODE_ATTRIBUTE);

			pnNextNode->AddRef();
			
			pnOutNextNode = pnNextNode;
			bResult = true;
		}
	}
	
	return bResult;
}


bool XMLDOM_Element::GetFirstChildNode(XMLDOM_Node *&pnOutFirstChild) const
{
	bool bResult = false;
	pnOutFirstChild = NULL;
	
	XMLDOM_InternalNodeStorage *pnsChildNodes = GetUnsafeChildNodes();

	size_type nChildCount = pnsChildNodes->GetLength();
	size_type nStartingChildIndex = GetStartingChildNodeIndex();
	if (nChildCount != nStartingChildIndex)
	{
		XMLDOM_Node *pnFirstChild = pnsChildNodes->GetUnsafeItem(nStartingChildIndex);
		assert(pnFirstChild->GetNodeType() != NODE_ATTRIBUTE);

		pnFirstChild->AddRef();

		pnOutFirstChild = pnFirstChild;
		bResult = true;
	}

	return bResult;
}

bool XMLDOM_Element::GetLastChildNode(XMLDOM_Node *&pnOutLastChild) const
{
	bool bResult = false;
	pnOutLastChild = NULL;
	
	XMLDOM_InternalNodeStorage *pnsChildNodes = GetUnsafeChildNodes();
	
	size_type nChildCount = pnsChildNodes->GetLength();
	size_type nStartingChildIndex = GetStartingChildNodeIndex();
	if (nChildCount != nStartingChildIndex)
	{
		XMLDOM_Node *pnLastChild = pnsChildNodes->GetUnsafeItem(nChildCount - 1);
		assert(pnLastChild->GetNodeType() != NODE_ATTRIBUTE);
		
		pnLastChild->AddRef();
		
		pnOutLastChild = pnLastChild;
		bResult = true;
	}
	
	return bResult;
}


void XMLDOM_Element::AppendChildNode(XMLDOM_Node *pnNewChild)
{
	assert(pnNewChild->GetUnsafeParentNode() == NULL); // Remove it from old parent first
	assert(pnNewChild->GetNodeType() != NODE_ATTRIBUTE && pnNewChild->GetNodeType() != NODE_DOCUMENT);

	XMLDOM_Document *pdOwnerDocument = GetUnsafeOwnerDocument();
	pnNewChild->SetOwnerDocument(pdOwnerDocument);

	pnNewChild->SetParentNode(this);

	XMLDOM_InternalNodeStorage *pnsChildNodes = GetUnsafeChildNodes();
	pnsChildNodes->AddItem(pnNewChild);
}

bool XMLDOM_Element::RemoveChildNode(XMLDOM_Node *pnOldChild)
{
	assert(pnOldChild->GetUnsafeParentNode() == this);
	assert(pnOldChild->GetNodeType() != NODE_ATTRIBUTE && pnOldChild->GetNodeType() != NODE_DOCUMENT);

	bool bResult = false;

	XMLDOM_InternalNodeStorage *pnsChildNodes = GetUnsafeChildNodes();
	size_type nStartingChildIndex = GetStartingChildNodeIndex();
	size_type nChildIndex = pnsChildNodes->GetItemIndex(pnOldChild, nStartingChildIndex, npos);

	if (nChildIndex != npos)
	{
		bResult = pnsChildNodes->RemoveItem(nChildIndex);

		pnOldChild->SetParentNode(NULL);
	}

	return bResult;
}

bool XMLDOM_Element::ReplaceChildNode(XMLDOM_Node *pnNewChild, XMLDOM_Node *pnOldChild)
{
	assert(pnNewChild != pnOldChild); // Senseless operation -- check before calling
	assert(pnNewChild->GetUnsafeParentNode() == NULL); // Remove it from old parent first
	assert(pnNewChild->GetNodeType() != NODE_ATTRIBUTE && pnNewChild->GetNodeType() != NODE_DOCUMENT);
	assert(pnOldChild->GetUnsafeParentNode() == this);
	assert(pnOldChild->GetNodeType() != NODE_ATTRIBUTE && pnOldChild->GetNodeType() != NODE_DOCUMENT);

	bool bResult = false;

	XMLDOM_InternalNodeStorage *pnsChildNodes = GetUnsafeChildNodes();
	size_type nStartingChildIndex = GetStartingChildNodeIndex();
	size_type nChildIndex = pnsChildNodes->GetItemIndex(pnOldChild, nStartingChildIndex, npos);

	if (nChildIndex != npos)
	{
		XMLDOM_Document *pdOwnerDocument = GetUnsafeOwnerDocument();
		pnNewChild->SetOwnerDocument(pdOwnerDocument);
		
		pnNewChild->SetParentNode(this);
		
		bResult = pnsChildNodes->ReplaceItem(nChildIndex, pnNewChild);

		pnOldChild->SetParentNode(NULL);
	}

	return bResult;
}

bool XMLDOM_Element::InsertChildNode(XMLDOM_Node *pnNewChild, XMLDOM_Node *pnOldChild)
{
	assert(pnNewChild != pnOldChild); // Senseless operation -- check before calling
	assert(pnNewChild->GetUnsafeParentNode() == NULL); // Remove it from old parent first
	assert(pnNewChild->GetNodeType() != NODE_ATTRIBUTE && pnNewChild->GetNodeType() != NODE_DOCUMENT);
	assert(pnOldChild->GetUnsafeParentNode() == this);
	assert(pnOldChild->GetNodeType() != NODE_ATTRIBUTE && pnOldChild->GetNodeType() != NODE_DOCUMENT);
	
	bool bResult = false;
	
	XMLDOM_InternalNodeStorage *pnsChildNodes = GetUnsafeChildNodes();
	size_type nStartingChildIndex = GetStartingChildNodeIndex();
	size_type nChildIndex = pnsChildNodes->GetItemIndex(pnOldChild, nStartingChildIndex, npos);
	
	if (nChildIndex != npos)
	{
		XMLDOM_Document *pdOwnerDocument = GetUnsafeOwnerDocument();
		pnNewChild->SetOwnerDocument(pdOwnerDocument);
		
		pnNewChild->SetParentNode(this);
		
		bResult = pnsChildNodes->InsertItem(nChildIndex, pnNewChild);
	}
	
	return bResult;
}


void XMLDOM_Element::RemoveChildNodes()
{
	XMLDOM_InternalNodeStorage *pnsChildNodes = GetUnsafeChildNodes();
	size_type nStartingChildIndex = GetStartingChildNodeIndex();

	pnsChildNodes->SetItemsParent(NULL, nStartingChildIndex, npos);
	pnsChildNodes->RemoveItems(nStartingChildIndex, npos);
}

void XMLDOM_Element::RemoveAttributesAndChildNodes()
{
	XMLDOM_InternalNodeStorage *pnsChildNodes = GetUnsafeChildNodes();
	
	pnsChildNodes->SetItemsParent(NULL);
	pnsChildNodes->RemoveItems();

	ResetAttributeNodeCount();
}


void XMLDOM_Element::GetElementsByTagName(const string &sTagNameOfInterest, XMLDOM_NodeList *&pnlOutResultNodes) const
{
	XMLDOM_NodeList *pnlResultNodes = new XMLDOM_NodeList();

	XMLDOM_InternalNodeStorage *pnsChildNodes = GetUnsafeChildNodes();

	size_type nChildCount = pnsChildNodes->GetLength();
	size_type nStartingChildIndex = GetStartingChildNodeIndex();
	for (size_type nChildIndex = nStartingChildIndex; nChildIndex != nChildCount; ++nChildIndex)
	{
		XMLDOM_Node *pnChildNode = pnsChildNodes->GetUnsafeItem(nChildIndex);
		assert(pnChildNode->GetNodeType() != NODE_ATTRIBUTE);

		DOMNodeType ntChildType = pnChildNode->GetNodeType();
		if (ntChildType == NODE_ELEMENT)
		{
			XMLDOM_Element *peChildElement = static_cast<XMLDOM_Element *>(pnChildNode);
			const string &sChildTag = peChildElement->GetTagName();
			if (sChildTag == sTagNameOfInterest)
			{
				pnlResultNodes->AddItem(pnChildNode);
			}
		}
	}

	pnlOutResultNodes = pnlResultNodes;
}

void XMLDOM_Element::SelectElementsByPattern(XMLDOM_Element *pePatternNode, XMLDOM_NodeList *&pnlOutResultNodes, bool bRecursive) const
{
	XMLDOM_NodeList *pnlResultNodes = new XMLDOM_NodeList();
	
	SelectElementsByPatternIntoList(pePatternNode, pnlResultNodes, bRecursive);
	
	pnlOutResultNodes = pnlResultNodes;
}

void XMLDOM_Element::SelectElementsByPatternIntoList(XMLDOM_Element *pePatternNode, XMLDOM_NodeList *pnlResultNodes, bool bRecursive) const
{
	XMLDOM_InternalNodeStorage *pnsChildNodes = GetUnsafeChildNodes();

	size_type nChildCount = pnsChildNodes->GetLength();
	size_type nStartingChildIndex = GetStartingChildNodeIndex();
	for (size_type nChildIndex = nStartingChildIndex; nChildIndex != nChildCount; ++nChildIndex)
	{
		XMLDOM_Node *pnChildNode = pnsChildNodes->GetUnsafeItem(nChildIndex);
		assert(pnChildNode->GetNodeType() != NODE_ATTRIBUTE);
		
		DOMNodeType ntChildType = pnChildNode->GetNodeType();
		if (ntChildType == NODE_ELEMENT)
		{
			XMLDOM_Element *peChildElement = static_cast<XMLDOM_Element *>(pnChildNode);

			if (GetIfElementsMatch(pePatternNode, peChildElement))
			{
				pnlResultNodes->AddItem(pnChildNode);
			}

			if (bRecursive)
			{
				peChildElement->SelectElementsByPatternIntoList(pePatternNode, pnlResultNodes, true);
			}
		}
	}
}

bool XMLDOM_Element::GetIfElementsMatch(XMLDOM_Element *pePatternNode, XMLDOM_Element *peTestElement)
{
	bool bResult = false;
	
	do
	{
		const string &sPatternTagName = pePatternNode->GetTagName();
		const string &sTestTagName = peTestElement->GetTagName();
		
		if (sPatternTagName != sTestTagName)
		{
			break;
		}

		XMLDOM_InternalNodeStorage *pnlPatternAttributes = pePatternNode->GetUnsafeChildNodes();
		
		size_type nPatternAttributeCount = pePatternNode->GetAttributeNodeCount();
		size_type nTestAttributeCount = peTestElement->GetAttributeNodeCount();

		if (nPatternAttributeCount != nTestAttributeCount)
		{
			break;
		}

		size_type nAttributeIndex = 0;
		for (; nAttributeIndex != nPatternAttributeCount; ++nAttributeIndex)
		{
			XMLDOM_Attr *paPatternAttribute = static_cast<XMLDOM_Attr *>(pnlPatternAttributes->GetUnsafeItem(nAttributeIndex));
			assert(paPatternAttribute->GetNodeType() == NODE_ATTRIBUTE);

			const string &sPatternAttributeName = paPatternAttribute->GetName();
			
			if (!peTestElement->GetAttributeExists(sPatternAttributeName))
			{
				break;
			}
		}
		if (nAttributeIndex != nPatternAttributeCount)
		{
			break;
		}

		string sPatternValue, sTestValue; 
		pePatternNode->GetNodeValue(sPatternValue);
		peTestElement->GetNodeValue(sTestValue);

		if (sPatternValue != sTestValue)
		{
			break;
		}
	
		bResult = true;
	}
	while (false);
	
	return bResult;
}


void XMLDOM_Element::SelectNodes(const string &sPatternString, XMLDOM_NodeList *&pnlOutResultList, bool bCaseInsensitive/* = false*/)
{
	XMLDOM_NodeList	*pnlResultList = new XMLDOM_NodeList();

	CXPath xpath(this, sPatternString, pnlResultList, CXPath::SELECT_ALL, bCaseInsensitive);
	xpath.Search();

	pnlOutResultList = pnlResultList;
}

bool XMLDOM_Element::SelectSingleNode(const string &sPatternString, XMLDOM_Node *&pnOutResultNode, bool bCaseInsensitive/* = false*/)
{
	bool bResult = false;
	pnOutResultNode = NULL;

	XMLDOM_NodeList	*pnlResultList = new XMLDOM_NodeList();
	
	CXPath xpath(this, sPatternString, pnlResultList, CXPath::SELECT_SINGLE, bCaseInsensitive);
	xpath.Search();

	size_type nNodeCount = pnlResultList->GetLength();
	if (nNodeCount != 0)
	{
		pnlResultList->GetItem(0, pnOutResultNode);

		bResult = true;
	}

	pnlResultList->Release();

	return bResult;
}


void XMLDOM_Element::GetTitle(string &sNodeTitle)
{
	sNodeTitle = GetTagName();
}

void XMLDOM_Element::GetText(string &sNodeText)
{
	GetNodeValue(sNodeText);
}

void XMLDOM_Element::PutText(const string &sNodeText)
{
	SetNodeValue(sNodeText);
}

void XMLDOM_Element::Clear()
{
	RemoveAttributesAndChildNodes();
}


string::size_type XMLDOM_Element::InternalGetXML(string &sBuffer, EncodingCharset nCharset) const
{
	string::size_type nSizeUsed = 0;
	string sChildBuffer;

	APPEND_SZ(sBuffer, nSizeUsed, NODE_TOKEN_STAG, STATIC_LENGTH(NODE_TOKEN_STAG));

	string sNamespace = GetNodeNamespace();
	string::size_type nNamespaceSize = sNamespace.size();
	if(nNamespaceSize > 0)
	{
		sNamespace.append(":");
		nNamespaceSize++;
		APPEND_STR(sBuffer, nSizeUsed, sNamespace, nNamespaceSize);
	}

	const string &sTagName = GetTagName();
	string::size_type nTagNameSize = sTagName.size();
	APPEND_STR(sBuffer, nSizeUsed, sTagName, nTagNameSize);

	XMLDOM_InternalNodeStorage *pnsChildNodes = GetUnsafeChildNodes();
	size_type nAttributeCount = GetAttributeNodeCount();

	{
		string::size_type nSizeUsedAtStart = nSizeUsed;
		size_type nItemCount = nAttributeCount;
		for (size_type nItemIndex = 0; nItemIndex != nItemCount; ++nItemIndex)
		{
			PREDICT_BUFFER(sBuffer, nSizeUsed, nSizeUsedAtStart, nItemIndex, nItemCount);

			XMLDOM_Attr *paElementAttr = static_cast<XMLDOM_Attr *>(pnsChildNodes->GetUnsafeItem(nItemIndex));
			string::size_type nChildSizeUsed = paElementAttr->InternalGetXML(sChildBuffer, nCharset);
			assert(paElementAttr->GetNodeType() == NODE_ATTRIBUTE);
			
			APPEND_STR(sBuffer, nSizeUsed, sChildBuffer, nChildSizeUsed);
		}
	}

	size_type nChildCount = pnsChildNodes->GetLength();
	size_type nStartingChildIndex = GetStartingChildNodeIndex();
	
	if (nChildCount == nStartingChildIndex)
	{
		APPEND_SZ(sBuffer, nSizeUsed, NODE_EMPTY_ETAG, STATIC_LENGTH(NODE_EMPTY_ETAG));
	}
	else 
	{
		APPEND_SZ(sBuffer, nSizeUsed, NODE_TOKEN_ETAG, STATIC_LENGTH(NODE_TOKEN_ETAG));

		string::size_type nSizeUsedAtStart = nSizeUsed;
		size_type nItemCount = nChildCount - nStartingChildIndex;
		for (size_type nItemIndex = 0; nItemIndex != nItemCount; ++nItemIndex)
		{
			PREDICT_BUFFER(sBuffer, nSizeUsed, nSizeUsedAtStart, nItemIndex, nItemCount);

			XMLDOM_Node *pnChildNode = pnsChildNodes->GetUnsafeItem(nItemIndex + nStartingChildIndex);
			string::size_type nChildSizeUsed = pnChildNode->InternalGetXML(sChildBuffer, nCharset);
			assert(pnChildNode->GetNodeType() != NODE_ATTRIBUTE);

			APPEND_STR(sBuffer, nSizeUsed, sChildBuffer, nChildSizeUsed);
		}

		APPEND_SZ(sBuffer, nSizeUsed, NODE_FINALISE_STAG, STATIC_LENGTH(NODE_FINALISE_STAG));
		if(nNamespaceSize > 0)
		{
			APPEND_STR(sBuffer, nSizeUsed, sNamespace, nNamespaceSize);
		}
		APPEND_STR(sBuffer, nSizeUsed, sTagName, nTagNameSize);
		APPEND_SZ(sBuffer, nSizeUsed, NODE_TOKEN_ETAG, STATIC_LENGTH(NODE_TOKEN_ETAG));
	}
	
	return nSizeUsed;
}

bool XMLDOM_Element::GetNamespaceByURI(const string& sURI, string& sNamespace) const
{
	bool	bRetv = false;
	string	sPattern, sNodeName, sNodeValue;
	size_t	nPos;
	XMLDOM_Element	*pElement, *pParentElement;
	XMLDOM_Node		*pNode = NULL;
	XMLDOM_NodeList	*pNodeList = NULL;
	
	sNamespace.erase();
	sPattern = "@*";
	pElement = (XMLDOM_Element*)this;
	
	while(pElement && !bRetv)
	{
		pElement->SelectNodes(sPattern, pNodeList);
		if(pNodeList && pNodeList->GetLength() > 0)
		{
			for(size_t i = 0; i < pNodeList->GetLength(); ++i)
			{
				if(!pNodeList->GetItem(i, pNode) || pNode == NULL)
				{
					continue;
				}
				sNodeName = pNode->GetNodeName();
				nPos = sNodeName.find(":");
				if(nPos != string::npos && nPos != sNodeName.size())
				{
					pNode->VirtualGetNodeValue(sNodeValue);
					if(sNodeValue == sURI)
					{
						sNamespace = sNodeName.substr(nPos + 1);
						bRetv = true;
						break;
					}
				}
			}
		}
		if(pElement->GetParentNode(pParentElement))
		{
			pElement = pParentElement;
		}
		else
		{
			pElement = NULL;
		}
	}
	return bRetv;
}



/************************************************************************/
/* XMLDOM_Document                                                      */
/************************************************************************/

#pragma warning(disable:4355)
XMLDOM_Document::XMLDOM_Document(): m_nCharSet(ECS_NONE),
	XMLDOM_Document_Parent(this, string())
{
}
#pragma warning(default:4355)

XMLDOM_Document::XMLDOM_Document(const XMLDOM_Document &dOther, bool bRecursive): m_nCharSet(ECS_NONE),
	XMLDOM_Document_Parent(dOther, bRecursive)
{
}

XMLDOM_Document::~XMLDOM_Document()
{
	// Do nothing
}


void XMLDOM_Document::CloneNode(XMLDOM_Node *&pnNewNode, bool bRecursive) const
{
	pnNewNode = new XMLDOM_Document(*this, bRecursive);
}


DOMNodeType XMLDOM_Document::GetNodeType() const
{
	return NODE_DOCUMENT;
}


void XMLDOM_Document::GetDocumentElement(XMLDOM_Element *&peOutDocumentElement) const
{
	XMLDOM_Element *peDocumentElement = GetUnsafeDocumentElement();
	
	if (peDocumentElement)
	{
		peDocumentElement->AddRef();
	}

	peOutDocumentElement = peDocumentElement;
}


XMLDOM_Element *XMLDOM_Document::GetUnsafeDocumentElement() const
{
	size_type nStartingChildIndex = GetStartingChildNodeIndex();
	XMLDOM_Element *peFirstElement = GetUnsafeFirstElementChildNode(nStartingChildIndex, NULL);
	return peFirstElement;
}


/*void XMLDOM_Document::CreateElement(const string &sTagName, XMLDOM_Element *&peElement)
{
	peElement = new XMLDOM_Element(this, sTagName);
}
*/
void XMLDOM_Document::CreateElement(const string &sTagName, const string &sTagNamespace, XMLDOM_Element *&peElement)
{
	peElement = new XMLDOM_Element(this, sTagName, sTagNamespace);
}

void XMLDOM_Document::CreateTextNode(const string &sContent, XMLDOM_Text *&ptText)
{
	ptText = new XMLDOM_Text(this, sContent);
}

void XMLDOM_Document::CreateComment(const string &sContent, XMLDOM_Comment *&pcComment)
{
	pcComment = new XMLDOM_Comment(this, sContent);
}

void XMLDOM_Document::CreateCDataSection(const string &sData, XMLDOM_CDATASection *&pdCData)
{
	pdCData = new XMLDOM_CDATASection(this, sData);
}

void XMLDOM_Document::CreateAttribute(const string &sAttributeName, XMLDOM_Attr *&paAttribute)
{
	paAttribute = new XMLDOM_Attr(this, sAttributeName);
}


bool XMLDOM_Document::LoadXML(const string &sXMLString, XMLDOM_FilterProc fpFilter, void *pvContext)
{
	CXMLParser xpParser;
	int iResult = xpParser.ParseDocument(sXMLString, (void *)fpFilter, pvContext, this);
	
#if defined(WIN32) || defined(WIN64)
	if (xpParser.GetLastError() != RESULT_OK)
	{
		string sOutputMessage = "Parse result: ";
		sOutputMessage += xpParser.ExplainLastError();
		sOutputMessage += "\n";
		//OutputDebugString(sOutputMessage.c_str());
	}
#endif // #if defined(WIN32) || defined(WIN64)

	return (iResult == RESULT_OK); 
}

bool XMLDOM_Document::LoadXMLEx(const string &sXMLString, string &sOutputMessage)
{
	CXMLParser xpParser;
	int iResult = xpParser.ParseDocument(sXMLString, NULL, NULL, this);
	
	sOutputMessage = xpParser.ExplainLastError();
	return (iResult == RESULT_OK); 
}

bool XMLDOM_Document::StringToNode(const string &sXMLString, XMLDOM_Node *&pnNewNode)
{
	CXMLParser xpParser;
	int iResult = xpParser.StringToNode(sXMLString, this, pnNewNode);

	return (iResult == RESULT_OK);
}

EncodingCharset XMLDOM_Document::GetCharset() const
{
	return (m_nCharSet == ECS_NONE) ? ECS_ASCII : m_nCharSet;
}

void XMLDOM_Document::SetCharset(EncodingCharset nCharset)
{
	m_nCharSet = nCharset;
	string sCharset = (nCharset == ECS_UTF8) ? szXMLPparser_UTF8_Encoding : "";
	if(GetAttributeNodeCount() == 0)
	{
		SetAttribute(szXMLPparser_AttrVersion, szXMLPparser_DefaultVersion);
	}
	SetAttribute(szXMLPparser_AttrEncoding, sCharset);
}

string::size_type XMLDOM_Document::InternalGetXML(string &sBuffer, EncodingCharset nCharset) const
{
	string::size_type nSizeUsed = 0;
	string sChildBuffer;
	bool bSetEncoding = false;
	EncodingCharset nDocCharset = (nCharset == ECS_NONE) ? m_nCharSet : nCharset;
	
	/*if(nDocCharset == ECS_UTF8)
	{
		APPEND_SZ(sBuffer, nSizeUsed, szXMLPparser_UTF8_BOM, STATIC_LENGTH(szXMLPparser_UTF8_BOM));
	}*/

	XMLDOM_InternalNodeStorage *pnsChildNodes = GetUnsafeChildNodes();
	size_type nAttributeCount = GetAttributeNodeCount();

	if (nAttributeCount != 0 || nDocCharset == ECS_UTF8)
	{
		APPEND_SZ(sBuffer, nSizeUsed, NODE_DOCDEF_STAG, STATIC_LENGTH(NODE_DOCDEF_STAG));

		string::size_type nSizeUsedAtStart = nSizeUsed;
		size_type nItemCount = nAttributeCount;
		for (size_type nItemIndex = 0; nItemIndex != nItemCount; ++nItemIndex)
		{
			PREDICT_BUFFER(sBuffer, nSizeUsed, nSizeUsedAtStart, nItemIndex, nItemCount);

			XMLDOM_Attr *paDocumentAttr = static_cast<XMLDOM_Attr *>(pnsChildNodes->GetUnsafeItem(nItemIndex));
			string::size_type nChildSizeUsed = 0;
			if( CompareStr(paDocumentAttr->GetName().c_str(), szXMLPparser_AttrEncoding) )
			{
				bSetEncoding = true;
				if(nCharset != ECS_NONE && nCharset != m_nCharSet)
				{
					XMLDOM_Attr* pnTempAttr;
					paDocumentAttr->CloneNode((XMLDOM_Node*&)pnTempAttr, false);
					pnTempAttr->SetValue(szXMLPparser_UTF8_Encoding);
					nChildSizeUsed = pnTempAttr->InternalGetXML(sChildBuffer, nCharset);
					assert(pnTempAttr->GetNodeType() == NODE_ATTRIBUTE);
					pnTempAttr->Release();
					continue;
				}
			}
			nChildSizeUsed = paDocumentAttr->InternalGetXML(sChildBuffer, nCharset);
			assert(paDocumentAttr->GetNodeType() == NODE_ATTRIBUTE);
			
			APPEND_STR(sBuffer, nSizeUsed, sChildBuffer, nChildSizeUsed);
		}
		string::size_type nChildSizeUsed = 0;
		if(nAttributeCount == 0)
		{
			XMLDOM_Attr* pnTempAttr;
			GetUnsafeOwnerDocument()->CreateAttribute(szXMLPparser_AttrVersion, pnTempAttr);
			pnTempAttr->SetValue(szXMLPparser_DefaultVersion);
			nChildSizeUsed = pnTempAttr->InternalGetXML(sChildBuffer, nCharset);
			pnTempAttr->Release();
			APPEND_STR(sBuffer, nSizeUsed, sChildBuffer, nChildSizeUsed);
		}
		if(!bSetEncoding && (nDocCharset != ECS_ASCII && nDocCharset != ECS_NONE))
		{
			XMLDOM_Attr* pnTempAttr;
			GetUnsafeOwnerDocument()->CreateAttribute(szXMLPparser_AttrEncoding, pnTempAttr);
			pnTempAttr->SetValue(szXMLPparser_UTF8_Encoding);
			nChildSizeUsed = pnTempAttr->InternalGetXML(sChildBuffer, nCharset);
			pnTempAttr->Release();
			APPEND_STR(sBuffer, nSizeUsed, sChildBuffer, nChildSizeUsed);
		}
		
		APPEND_SZ(sBuffer, nSizeUsed, NODE_DOCDEF_ETAG, STATIC_LENGTH(NODE_DOCDEF_ETAG));
	}

	size_type nChildCount = pnsChildNodes->GetLength();
	size_type nStartingChildIndex = GetStartingChildNodeIndex();

	{
		string::size_type nSizeUsedAtStart = nSizeUsed;
		size_type nItemCount = nChildCount - nStartingChildIndex;
		for (size_type nItemIndex = 0; nItemIndex != nItemCount; ++nItemIndex)
		{
			PREDICT_BUFFER(sBuffer, nSizeUsed, nSizeUsedAtStart, nItemIndex, nItemCount);

			XMLDOM_Node *pnChildNode = pnsChildNodes->GetUnsafeItem(nItemIndex + nStartingChildIndex);
			string::size_type nChildSizeUsed = pnChildNode->InternalGetXML(sChildBuffer, nCharset);
			assert(pnChildNode->GetNodeType() != NODE_ATTRIBUTE);

			APPEND_STR(sBuffer, nSizeUsed, sChildBuffer, nChildSizeUsed);
		}
	}

	return nSizeUsed;
}


//////////////////////////////////////////////////////////////////////////

static const char szWhiteSpaces[] = "\t\n\r ";
static const int nWhiteSpacesLen = sizeof( szWhiteSpaces ) - 1;

string TrimWhiteSpaces(const string& value)
{
	string sResult;

	if (!value.empty())
	{
		string::size_type nBegin = value.find_first_not_of(szWhiteSpaces);
		string::size_type nEnd = value.find_last_not_of(szWhiteSpaces);
		if (nEnd != string::npos)
		{
			++nEnd;
		}
		
		sResult.assign(value, nBegin, nEnd - nBegin);
	}

	return sResult;
}
