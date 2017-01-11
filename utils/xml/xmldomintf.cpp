#include "StdAfx.h"
#include "xmldomintf.h"

/*
 * CXMLDOMBaseImpl
 */

CXMLDOMBaseImpl::CXMLDOMBaseImpl(const CXMLDOMBaseImpl &biOther):
	m_pImpl(biOther.m_pImpl)
{
	XMLDOM_Base *pOwnImpl = m_pImpl;
	if (pOwnImpl)
	{
		pOwnImpl->AddRef();
	}
}

CXMLDOMBaseImpl::~CXMLDOMBaseImpl()
{
	XMLDOM_Base *pOwnImpl = m_pImpl;
	if (pOwnImpl)
	{
		pOwnImpl->Release();
	}
}

CXMLDOMBaseImpl &CXMLDOMBaseImpl::operator =(const CXMLDOMBaseImpl &biOther)
{
	XMLDOM_Base *pOwnImpl = m_pImpl;
	XMLDOM_Base *pOtherImpl = biOther.m_pImpl;
	
	if (pOwnImpl != pOtherImpl)
	{
		if (pOtherImpl)
		{
			pOtherImpl->AddRef();
		}
		
		m_pImpl = pOtherImpl;
		
		if (pOwnImpl)
		{
			pOwnImpl->Release();
		}
	}

	return *this;
}

CXMLDOMBaseImpl &CXMLDOMBaseImpl::operator =(const void *pOfNULL)
{
	assert(pOfNULL == NULL);

	XMLDOM_Base *pOwnImpl = m_pImpl;
	
	if (pOwnImpl != NULL)
	{
		m_pImpl = NULL;
		
		pOwnImpl->Release();
	}
	
	return *this;
}


/**************************************************************************************************************

													CXMLDOMNodeList

***************************************************************************************************************/



unsigned int CXMLDOMNodeList::getLength() const
{
	unsigned int uiResult = 0;

	XMLDOM_NodeList *pImpl = GetImpl();
	if (pImpl)
	{
		uiResult = (unsigned int)pImpl->GetLength();
	}

	return uiResult;
}


CXMLDOMNode CXMLDOMNodeList::item(unsigned int index) const
{
	XMLDOM_NodeList *pImpl = GetImpl();

	XMLDOM_Node *pnItem;
	bool bGetResult = pImpl->GetItem(index, pnItem);
	assert(bGetResult && pnItem);

	return CXMLDOMNode(pnItem);
}

void CXMLDOMNodeList::erase(unsigned int index)
{
	XMLDOM_NodeList *pImpl = GetImpl();
	
	bool bEraseResult = pImpl->RemoveItem(index);
	assert(bEraseResult);
}


CXMLDOMNodeList CXMLDOMNodeList::cloneNodes(bool deep)
{
	XMLDOM_NodeList *pnlNewList = new XMLDOM_NodeList();
	
	XMLDOM_NodeList *pImpl = GetImpl();

	unsigned int nItemCount = (unsigned int)pImpl->GetLength();
	for (unsigned int nItemIndex = 0; nItemIndex != nItemCount; ++nItemIndex)
	{
		XMLDOM_Node *pnExistingNode;
		if (pImpl->GetItem(nItemIndex, pnExistingNode))
		{
			XMLDOM_Node *pnNewNode;
			pnExistingNode->CloneNode(pnNewNode, deep);

			pnlNewList->AddItem(pnNewNode);

			pnNewNode->Release();
			pnExistingNode->Release();
		}
	}//for

	return CXMLDOMNodeList(pnlNewList);
}

/**************************************************************************************************************

													CXMLDOMNode

***************************************************************************************************************/


int CXMLDOMNode::getNodeType() const
{
	XMLDOM_Node *pImpl = GetImpl();
    return pImpl->GetNodeType();
}


CXMLDOMElement CXMLDOMNode::getParentNode() const
{
	XMLDOM_Element *pParentImpl;

	XMLDOM_Node *pImpl = GetImpl();
	pImpl->GetParentNode(pParentImpl);

	return CXMLDOMElement(pParentImpl);
}

CXMLDOMDocument CXMLDOMNode::getOwnerDocument() const
{
	XMLDOM_Document *pDocumentImpl;
	
	XMLDOM_Node *pImpl = GetImpl();
	pImpl->GetOwnerDocument(pDocumentImpl);
	
    return CXMLDOMDocument(pDocumentImpl);
}


CXMLDOMNode CXMLDOMNode::cloneNode(bool deep) const 
{
	XMLDOM_Node *pCloneImpl;
	
	XMLDOM_Node *pImpl = GetImpl();
	pImpl->CloneNode(pCloneImpl, deep);
	
    return CXMLDOMNode(pCloneImpl);
}


string CXMLDOMNode::getNodeName() const 
{
	XMLDOM_Node *pImpl = GetImpl();
	return pImpl->VirtualGetNodeName();
}

string CXMLDOMNode::getNodeValue() const
{
	string sResult;

	XMLDOM_Node *pImpl = GetImpl();
	pImpl->VirtualGetNodeValue(sResult);

	return sResult;
}

void CXMLDOMNode::setNodeName(const string &nodeName)
{
	XMLDOM_Node *pImpl = GetImpl();
	pImpl->VirtualSetNodeName(nodeName);
}

void CXMLDOMNode::setNodeValue(const string &nodeValue)
{
    XMLDOM_Node *pImpl = GetImpl();
	pImpl->VirtualSetNodeValue(nodeValue);
}


bool CXMLDOMNode::hasChildNodes() const
{
    XMLDOM_Node *pImpl = GetImpl();
    return pImpl->VirtualHasChildNodes();
}

CXMLDOMNodeList CXMLDOMNode::getChildNodes() const
{
	XMLDOM_NodeList *pnlChildNodes;

    XMLDOM_Node *pImpl = GetImpl();
	pImpl->VirtualGetChildNodes(pnlChildNodes);

	return CXMLDOMNodeList(pnlChildNodes);
}

void CXMLDOMNode::removeChildNodes()
{
    XMLDOM_Node *pImpl = GetImpl();
	pImpl->VirtualRemoveChildNodes();
}


CXMLDOMNode CXMLDOMNode::getFirstChild() const
{
	XMLDOM_Node *pnChildNode;
	
    XMLDOM_Node *pImpl = GetImpl();
	pImpl->VirtualGetFirstChildNode(pnChildNode);
	
	return CXMLDOMNode(pnChildNode);
}


CXMLDOMNode CXMLDOMNode::getLastChild() const
{
	XMLDOM_Node *pnChildNode;
	
    XMLDOM_Node *pImpl = GetImpl();
	pImpl->VirtualGetLastChildNode(pnChildNode);
	
	return CXMLDOMNode(pnChildNode);
}


CXMLDOMNode CXMLDOMNode::getPreviousSibling() const
{
	XMLDOM_Node *pnChildNode;
	
    XMLDOM_Node *pImpl = GetImpl();
	pImpl->VirtualGetPreviousSiblingNode(pnChildNode);
	
	return CXMLDOMNode(pnChildNode);
}


CXMLDOMNode CXMLDOMNode::getNextSibling() const
{
	XMLDOM_Node *pnChildNode;
	
    XMLDOM_Node *pImpl = GetImpl();
	pImpl->VirtualGetNextSiblingNode(pnChildNode);
	
	return CXMLDOMNode(pnChildNode);
}

int CXMLDOMNode::getIndexInParent() const
{
    XMLDOM_Node *pImpl = GetImpl();
	return (int)pImpl->VirtualGetIndexInParent();
}

void CXMLDOMNode::removeFromParent()
{
    XMLDOM_Node *pImpl = GetImpl();
	pImpl->RemoveFromParent();
}


CXMLDOMNode CXMLDOMNode::insertBefore(const CXMLDOMNode &newChild, const CXMLDOMNode &refChild)
{
	CXMLDOMNode resultChild;

    XMLDOM_Node *pImpl = GetImpl();
	if (pImpl->VirtualInsertChildNode(newChild.GetImpl(), refChild.GetImpl()))
	{
		resultChild = newChild;
	}

    return resultChild;
}

CXMLDOMNode CXMLDOMNode::replaceChild(const CXMLDOMNode &newChild, const CXMLDOMNode &oldChild)
{
	CXMLDOMNode resultChild;

    XMLDOM_Node *pImpl = GetImpl();
	if (pImpl->VirtualReplaceChildNode(newChild.GetImpl(), oldChild.GetImpl()))
	{
		resultChild = oldChild;
	}
	
    return resultChild;
}

CXMLDOMNode CXMLDOMNode::removeChild(const CXMLDOMNode &oldChild)
{
	CXMLDOMNode resultChild;
	
    XMLDOM_Node *pImpl = GetImpl();
	if (pImpl->VirtualRemoveChildNode(oldChild.GetImpl()))
	{
		resultChild = oldChild;
	}
	
    return resultChild;
}

CXMLDOMNode CXMLDOMNode::appendChild(const CXMLDOMNode &newChild)
{
    XMLDOM_Node *pImpl = GetImpl();
	pImpl->VirtualAppendChildNode(newChild.GetImpl());
    
	return newChild;
}
 
  
void CXMLDOMNode::clear()
{
	XMLDOM_Node *pImpl = GetImpl();
	pImpl->Clear();
}

void CXMLDOMNode::normalize()
{
    // Not implemented
}


string CXMLDOMNode::get_xml(EncodingCharset nCharset) const
{
	string sBuffer;

    XMLDOM_Node *pImpl = GetImpl();
	pImpl->GetXML(sBuffer, nCharset);

	return sBuffer;
}

void CXMLDOMNode::get_xml(string & sXML, EncodingCharset nCharset) const
{
	XMLDOM_Node *pImpl = GetImpl();
	pImpl->GetXML(sXML, nCharset);
}

string CXMLDOMNode::get_title() const
{
	string sBuffer;
	
    XMLDOM_Node *pImpl = GetImpl();
	pImpl->GetTitle(sBuffer);
	
	return sBuffer;
}

string CXMLDOMNode::get_text() const
{
	string sBuffer;
	
    XMLDOM_Node *pImpl = GetImpl();
	pImpl->GetText(sBuffer);

	return sBuffer;
}

void CXMLDOMNode::put_text(const string &text)
{
    XMLDOM_Node *pImpl = GetImpl();
	pImpl->PutText(text);
}


CXMLDOMNodeList CXMLDOMNode::selectNodesByPattern(const CXMLDOMNode &PatternNode, bool deep)
{
	XMLDOM_NodeList *pnlResultList = NULL;

	if (PatternNode.getNodeType() == NODE_ELEMENT)
	{
		XMLDOM_Node *pImpl = GetImpl();
		pImpl->VirtualSelectElementsByPattern(static_cast<XMLDOM_Element *>(PatternNode.GetImpl()), pnlResultList, deep);
	}

	return CXMLDOMNodeList(pnlResultList);
}

CXMLDOMNodeList CXMLDOMNode::selectNodes(const string &patternString, bool bCaseInsensitive) const 
{
	XMLDOM_NodeList *pnlResultList = NULL;
	
	if(!patternString.empty())
	{
		XMLDOM_Node *pImpl = GetImpl();
		pImpl->VirtualSelectNodes(patternString, pnlResultList, bCaseInsensitive);
	}

    return CXMLDOMNodeList(pnlResultList);
}

CXMLDOMNode CXMLDOMNode::selectSingleNode(const string &patternString, bool bCaseInsensitive) const
{
	XMLDOM_Node *pnResultNode = NULL;
	
	if(!patternString.empty())
	{
		XMLDOM_Node *pImpl = GetImpl();
		pImpl->VirtualSelectSingleNode(patternString, pnResultNode, bCaseInsensitive);
	}
	
    return CXMLDOMNode(pnResultNode);
}




/**************************************************************************************************************

													CXMLDOMAttr

***************************************************************************************************************/


string CXMLDOMAttr::getName() const
{
	XMLDOM_Attr *pImpl = GetImpl();
    return pImpl->GetName();
}

string CXMLDOMAttr::getValue() const
{
	XMLDOM_Attr *pImpl = GetImpl();
    return pImpl->GetValue();
}

void CXMLDOMAttr::setValue(const string &value) 
{
	XMLDOM_Attr *pImpl = GetImpl();
	pImpl->SetValue(value);
}



/**************************************************************************************************************

													CXMLDOMCharacterData

***************************************************************************************************************/




string CXMLDOMCharacterData::getData() const
{
	XMLDOM_CharacterData *pImpl = GetImpl();
	return pImpl->GetData();
}


void CXMLDOMCharacterData::setData(const string &data)
{
	XMLDOM_CharacterData *pImpl = GetImpl();
	pImpl->SetData(data);
}


string CXMLDOMCharacterData::substringData(unsigned int offset, unsigned int count) const
{
	XMLDOM_CharacterData *pImpl = GetImpl();
	const string &sData = pImpl->GetData();
	
	return string(sData, offset, count);
}


void CXMLDOMCharacterData::appendData(const string &arg)
{
	XMLDOM_CharacterData *pImpl = GetImpl();
	pImpl->AppendData(arg);
}

void CXMLDOMCharacterData::insertData(unsigned int offset, const string &arg)
{
	XMLDOM_CharacterData *pImpl = GetImpl();
	pImpl->InsertData(offset, arg);
}

void CXMLDOMCharacterData::deleteData(unsigned int offset, unsigned int count)
{
	XMLDOM_CharacterData *pImpl = GetImpl();
	pImpl->DeleteData(offset, count);
}

void CXMLDOMCharacterData::replaceData(unsigned int offset, unsigned int count, const string &arg)
{
	XMLDOM_CharacterData *pImpl = GetImpl();
	pImpl->ReplaceData(offset, count, arg);
}


/**************************************************************************************************************

													CXMLDOMText

***************************************************************************************************************/



/**************************************************************************************************************

													CXMLDOMCDATASection

***************************************************************************************************************/



/**************************************************************************************************************

													CXMLDOMComment

***************************************************************************************************************/



/**************************************************************************************************************

													CXMLDOMElement

***************************************************************************************************************/


string CXMLDOMElement::getTagName() const
{
	XMLDOM_Element *pImpl = GetImpl();
	return pImpl->GetTagName();
}

string CXMLDOMElement::getNamespace() const
{
	XMLDOM_Element *pImpl = GetImpl();
	return pImpl->GetNamespace();
}

bool CXMLDOMElement::getNamespaceByURI(const string& sURI, string& sNamespace) const
{
	XMLDOM_Element *pImpl = GetImpl();
	return pImpl->GetNamespaceByURI(sURI, sNamespace);
}

string CXMLDOMElement::getAttribute(const string &name) const
{
	string sAttributeValue;

	XMLDOM_Element *pImpl = GetImpl();
	pImpl->GetAttribute(name, sAttributeValue);

	return sAttributeValue;
}

string CXMLDOMElement::getAttribute(const string &name, bool &bRes) const
{
	string sAttributeValue;
	
	XMLDOM_Element *pImpl = GetImpl();
	bRes = pImpl->GetAttribute(name, sAttributeValue);
	
	return sAttributeValue;
}

void CXMLDOMElement::setAttribute(const string &name, const string &value)
{
	XMLDOM_Element *pImpl = GetImpl();
	pImpl->SetAttribute(name, value);
}

void CXMLDOMElement::removeAttribute(const string &name)
{
	XMLDOM_Element *pImpl = GetImpl();
	pImpl->RemoveAttribute(name);
}


CXMLDOMAttr  CXMLDOMElement::getAttributeNode(const string &name) const
{
	XMLDOM_Attr *pAttribue;

	XMLDOM_Element *pImpl = GetImpl();
	pImpl->GetAttributeNode(name, pAttribue);
    
	return CXMLDOMAttr(pAttribue);
}

CXMLDOMAttr CXMLDOMElement::setAttributeNode(const CXMLDOMAttr &newAttr)
{
	XMLDOM_Attr *pAttrImpl;

	XMLDOM_Element *pImpl = GetImpl();
	pImpl->SetAttributeNode(newAttr.GetImpl(), pAttrImpl);

	return CXMLDOMAttr(pAttrImpl);
}

CXMLDOMAttr CXMLDOMElement::removeAttributeNode(const CXMLDOMAttr &oldAttr)
{
	XMLDOM_Attr *pAttrImpl;
	
	XMLDOM_Element *pImpl = GetImpl();
	pImpl->RemoveAttributeNode(oldAttr.GetImpl(), pAttrImpl);
	
	return CXMLDOMAttr(pAttrImpl);
}


bool CXMLDOMElement::getAttributesList( CXMLDOMAttributesList &xdalAttribs ) const
{
	xdalAttribs.clear();

	XMLDOM_NodeList *pnlAttrList;
	
	XMLDOM_Element *pImpl = GetImpl();
	pImpl->GetAttributes(pnlAttrList);

	XMLDOM_NodeList::size_type nAttributeCount = pnlAttrList->GetLength();
	for (XMLDOM_NodeList::size_type nAttributeIndex = 0; nAttributeIndex != nAttributeCount; ++nAttributeIndex)
	{
		XMLDOM_Node *pnAttribute;
		pnlAttrList->GetItem(nAttributeIndex, pnAttribute);
		assert(pnAttribute->GetNodeType() == NODE_ATTRIBUTE);

		xdalAttribs.push_back(CXMLDOMAttr(static_cast<XMLDOM_Attr *>(pnAttribute)));
	}

	pnlAttrList->Release();
	
	return nAttributeCount != 0;
}

CXMLDOMNodeList CXMLDOMElement::getElementsByTagName(const string &name) const
{
	XMLDOM_NodeList *pnlElements;

	XMLDOM_Element *pImpl = GetImpl();
	pImpl->GetElementsByTagName(name, pnlElements);

	return CXMLDOMNodeList(pnlElements);
}


/**************************************************************************************************************

													CXMLDOMDocument

***************************************************************************************************************/


CXMLDOMDocument CXMLDOMDocument::createDocument()
{
	XMLDOM_Document *pdDocument = new XMLDOM_Document();
	return CXMLDOMDocument(pdDocument);
}


CXMLDOMElement CXMLDOMDocument::createElement(const string &tagName, 
	const string &tagNamespace /* =  */) const
{
	XMLDOM_Element *peElement;

	XMLDOM_Document *pImpl = GetImpl();
	pImpl->CreateElement(tagName, tagNamespace, peElement);
	
	return CXMLDOMElement(peElement);
}


CXMLDOMText CXMLDOMDocument::createTextNode(const string &data) const
{
	XMLDOM_Text *ptText;

	XMLDOM_Document *pImpl = GetImpl();
	pImpl->CreateTextNode(data, ptText);

	return CXMLDOMText(ptText);
}


CXMLDOMComment CXMLDOMDocument::createComment(const string &data) const
{
	XMLDOM_Comment *pcComment;

	XMLDOM_Document *pImpl = GetImpl();
	pImpl->CreateComment(data, pcComment);

	return CXMLDOMComment(pcComment);
}


CXMLDOMCDATASection CXMLDOMDocument::createCDATASection(const string &data) const
{
	XMLDOM_CDATASection *pcdCData;

	XMLDOM_Document *pImpl = GetImpl();
	pImpl->CreateCDataSection(data, pcdCData);

	return CXMLDOMCDATASection(pcdCData);
}

CXMLDOMAttr CXMLDOMDocument::createAttribute(const string &name) const
{
	XMLDOM_Attr *paAttriibute;

	XMLDOM_Document *pImpl = GetImpl();
	pImpl->CreateAttribute(name, paAttriibute);

	return CXMLDOMAttr(paAttriibute);
}


CXMLDOMElement CXMLDOMDocument::getDocumentElement() const 
{
	XMLDOM_Element *peElement;
	
	XMLDOM_Document *pImpl = GetImpl();
	pImpl->GetDocumentElement(peElement);

	return CXMLDOMElement(peElement);
}


CXMLDOMNode CXMLDOMDocument::StringToNode(const string &from)
{
	XMLDOM_Node *pnNode;

	XMLDOM_Document *pImpl = GetImpl();
	pImpl->StringToNode(from, pnNode);

	return CXMLDOMNode(pnNode);
}


bool CXMLDOMDocument::loadXML(const string &strxml)
{
	XMLDOM_Document *pImpl = GetImpl();
	return pImpl->LoadXML(strxml, NULL, NULL);
}

bool CXMLDOMDocument::loadXMLEx(const string &strxml, string &strerr)
{
	XMLDOM_Document *pImpl = GetImpl();
	return pImpl->LoadXMLEx(strxml, strerr);
}

bool CXMLDOMDocument::loadXMLFiltered(const string &strxml, XMLDOM_FilterProc fpFilter, void *pvContext)
{
	XMLDOM_Document *pImpl = GetImpl();
	return pImpl->LoadXML(strxml, fpFilter, pvContext);
}

EncodingCharset CXMLDOMDocument::getCharset() const
{
	XMLDOM_Document *pImpl = GetImpl();
	return pImpl->GetCharset();
}

void CXMLDOMDocument::setCharset(EncodingCharset nCharset)
{
	XMLDOM_Document *pImpl = GetImpl();
	pImpl->GetCharset();
}


//////////////////////////////////////////////////////////////////////////
// NormalizeXMLDOMDocument

string GetCorrectXMLString( CXMLDOMNode xmlNode, int nLevel )
{
	const int nCountSpace = 1; 
	string sResult, sChildText, sLevelSp;
	
	if( xmlNode.isNull() ) 
	{
		return string();
	}

	CXMLDOMNodeList xmlNodeList;
	CXMLDOMAttributesList xmlAttrList;
	CXMLDOMNode xmlChildNode;

	int nCountSpaces = nLevel * nCountSpace;
	sLevelSp.resize(nCountSpaces, (char)'\t');
	
	switch( xmlNode.getNodeType() ) 
	{
		case NODE_DOCUMENT: // Contains the literal string "#comment".
		{
			xmlNodeList = xmlNode.getChildNodes();
		
			for( unsigned int i = 0; i < xmlNodeList.getLength(); i++ )
			{
				sResult += GetCorrectXMLString( xmlNodeList.item( i ), 1 );
				sResult += "\r\n"; 
			}

			break;
		}
		case NODE_ELEMENT: //Contains the name of the XML tag, with any namespace prefix included if present. 
		{
			sResult = sLevelSp + string("<") + xmlNode.getNodeName();
			(( CXMLDOMElement& )xmlNode ).getAttributesList( xmlAttrList );

			for( unsigned int i = 0; i < xmlAttrList.size(); i++ )
			{
				sResult += xmlAttrList[ i ].get_xml();
			}

			xmlNodeList = xmlNode.getChildNodes();
			if( !xmlNodeList.isNull() && 
				xmlNodeList.getLength() == 1 && 
				xmlNodeList.item( 0 ).getNodeType() == NODE_TEXT )
			{
				sResult += ">";
				xmlChildNode = xmlNodeList.item( 0 );
				sResult += xmlChildNode.get_xml() + "</" + xmlNode.getNodeName() + ">";
			}
			else
			{
				if( !xmlNodeList.isNull() && 
					xmlNodeList.getLength() > 0 )
				{
					sResult += ">";
					
					for( unsigned int i = 0; i < xmlNodeList.getLength(); i++ )
					{
						CXMLDOMNode xmlChildNode = xmlNodeList.item( i );
						sChildText = GetCorrectXMLString( xmlChildNode, nLevel + 1 );
						sResult += "\r\n" + sChildText;
					}

					sResult += "\r\n";
					sResult.resize(sResult.size() + nCountSpaces, (char)'\t');
					sResult += "</" + xmlNode.getNodeName() + ">";
				}
				else
				{
					sResult += "/>";
				}
				
			}

			break;
		}
		case NODE_ATTRIBUTE:
		{
			sResult +=  xmlNode.getNodeValue();
			break;
		}
		default:
		{
			assert(false);
			// break; -- proceed to case NODE_TEXT
		}
		case NODE_CDATA_SECTION: // Contains the literal string "#cdata-section".
		case NODE_COMMENT: // Contains the literal string "#comment".
		case NODE_TEXT: // Contains the literal string "#text".
		{
			sResult = sLevelSp + xmlNode.get_xml();
			break;
		}
	}//switch

	return sResult;
}

string NormalizeXMLDOMDocument( CXMLDOMNode xmlNode )
{
	string sResult;
	sResult = GetCorrectXMLString( xmlNode, 0 );
	return sResult;
}
