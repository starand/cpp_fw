// XPath.cpp: implementation of the CXPath class.
//
//////////////////////////////////////////////////////////////////////

#include "StdAfx.h"
#include "FilterExpr.h"
#include "strutils.h"
#include "xpath.h"
#include "strutils.h"

using namespace StrUtils;

extern void TrimAll(string& sTemp, char chLeft=' ', char chRight=' ');

inline void TrimStr(string& sTemp)
{
	TrimAll(sTemp);
}

void TrimBrStr(string& sTemp)
{
	TrimAll(sTemp, '(', ')');
}

void RemoveOrphanBrackets(string& sQuery)
{
	while (1)
	{
		int nBrCounter = 0;
		
		for (string::size_type pos = 0; pos < sQuery.size(); ++pos)
		{
			char ch = sQuery[pos];
			
			if (pos == 0 && ch != '(')
			{
				break;
			}//if
			
			switch(ch)
			{
			case '(':
				++nBrCounter;
				break;
				
			case ')':
				--nBrCounter;
				
				if (nBrCounter == 0)
				{
					//check if last element
					if (pos + 1 >= sQuery.size())
					{
						sQuery.erase(0, 1);
						sQuery.erase(sQuery.size() - 1);
						
						continue;
					}//if	
					
					return;
				}//if
				
				break;
			}//switch
		}//for
		
		break;
	}//while
}



//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////



void QUERYITEM::AddName(const string& asName1)
{
    string asName = asName1;
//	if ((res = asName.find("(", 0)) != -1)
//		asName.erase(res, 1);
//	while ((res = asName.find(")", 0)) != -1)
//		asName = asName.erase(res, 1);
	nodeType = NODE_INVALID;
	string::size_type i = 0;
	int startpos = 0;

	while (i < asName.size()) 
	{
		switch(asName[i])
		{
		case '(':
			string func = asName.substr(startpos, i - startpos);

			if (!func.compare("ancestor"))
			{
				string::size_type iRes = asName.find(")", i);
				if (iRes == string::npos) iRes = asName.size() + 1;
				name = asName.substr(i + 1, iRes - i - 1);
				findType = _ANCESTOR;
				nodeType = NODE_ELEMENT;
			}

			if (!func.compare("attribute"))
			{
				string::size_type iRes = asName.find(")", i);
				if (iRes == string::npos) iRes = asName.size() + 1;
				name = asName.substr(i + 1, iRes - i - 1);
				nodeType = NODE_ATTRIBUTE;
			}

			if (!func.compare("element"))
			{
				string::size_type iRes = asName.find(")", i);
				if (iRes == string::npos) iRes = asName.size() + 1;
				name = asName.substr(i + 1, iRes - i - 1);
				nodeType = NODE_ELEMENT;
			}

			if (!func.compare("cdata")) nodeType = NODE_CDATA_SECTION;
			if (!func.compare("comment")) nodeType = NODE_COMMENT;
			if (!func.compare("textnode")) nodeType = NODE_TEXT;
			if (!func.compare("text")) nodeType = NODE_TEXT_CDATA;
			if (!func.compare("node")) nodeType = NODE_NODE;

			break;
		}
		
		++i;
	}

	if (nodeType == NODE_INVALID) 
	{
		TrimBrStr(asName);

		if (!asName.empty() && asName[0]=='@') 
		{
			name = asName.substr(1);
			nodeType = NODE_ATTRIBUTE; 
		}
		else
		{
			name = asName;
			nodeType = NODE_ELEMENT;
		}
	}
}

void QUERYITEM::ItemClear()
{
	name.erase();
	Condition.clear();
	findType = _UNDEF;
	nodeType = NODE_INVALID;
}

CXPath::CXPath(XMLDOM_Node *pNode, const string& sQuery, XMLDOM_NodeList *foundList, int iNumber, bool bCaseInsensitive) :
	m_StartNode(pNode), m_FoundList(foundList), m_QueryString(sQuery), m_Number(iNumber), m_bCaseInsensitive(bCaseInsensitive)
{
}

CXPath::~CXPath()
{

}

void CXPath::Search()
{
	try
	{
		ExecuteQuery(m_QueryString, NULL);
	}
	catch(...)
	{
		return;
	}
}


QUERYRESULT CXPath::ExecuteQuery(string& sPattern, XMLDOM_Node *pStartNode)
{
	QUERYRESULT qrResult = _FALSE;

	vector<int> vecIndex;

	do
	{
		list<QUERYITEM> listQueryItem;
		if (!ParseQuery(sPattern, listQueryItem))
		{
			if (pStartNode)
			{
				m_FoundList->AddItem(pStartNode);
				
				qrResult = CheckEnding();
			}//if
			
			break;
		}
		
		list<QUERYITEM>::iterator itBegin = listQueryItem.begin(), itEnd = listQueryItem.end();
		
		list<QUERYITEM>::iterator itNext = itBegin; 
		if (itNext != itEnd)
		{
			++itNext;
			
			if (itNext != itEnd)
			{
				for (list<QUERYITEM>::iterator it = itBegin; it != itEnd; ++it)
				{
					QUERYITEM &QueryItem = *it;
					
					if (ExecuteQuery(QueryItem.sNextQuery, pStartNode) == _ENDING)
					{
						qrResult = _ENDING;
						break;
					}
				}

				break;
			}

			QUERYITEM &QueryItem = *itBegin;
			qrResult = ExecuteSingleQueryItem(pStartNode, QueryItem, vecIndex);
		}
	}
	while (false);
	
	return qrResult;
}

QUERYRESULT CXPath::ExecuteSingleQueryItem(XMLDOM_Node *pStartNode, QUERYITEM &QueryItem, vector<int> &vecIndex)
{
	QUERYRESULT qrResult = _FALSE;

	XMLDOM_Node *pCurNode;
	bool bSkipMovingDown;
	if (DefineStart(pStartNode, pCurNode, bSkipMovingDown, QueryItem)) 
	{
		if (QueryItem.name.compare(".") == 0)
		{
			qrResult = ExecuteQuery(QueryItem.sNextQuery, pCurNode);
		}
		else if (QueryItem.name.compare("..") == 0)
		{
			XMLDOM_Node *pUpperNode;
			if (MoveUp(pCurNode, pUpperNode))
			{
				qrResult = ExecuteQuery(QueryItem.sNextQuery, pUpperNode);
				
				pUpperNode->Release();
			}
		}
		else if (QueryItem.findType == QUERYITEM::_ANCESTOR) 
		{
			qrResult = ExecuteQuery(QueryItem.sNextQuery, pCurNode);
		}
		else if (bSkipMovingDown)
		{
			qrResult = ExecuteQueryItemForLevelNodes(pCurNode, QueryItem, vecIndex);
		}
		else
		{
			XMLDOM_Node *pLowerNode;
			if (MoveDown(pCurNode, pLowerNode, QueryItem)) 
			{
				qrResult = ExecuteQueryItemForLevelNodes(pLowerNode, QueryItem, vecIndex);
				
				pLowerNode->Release();
			}
		}
		
		pCurNode->Release();
	}

	return qrResult;
}

QUERYRESULT CXPath::ExecuteQueryItemForLevelNodes(XMLDOM_Node *pStartNode, QUERYITEM &QueryItem, vector<int> &vecIndex)
{
	QUERYRESULT qrResult = _FALSE;

	XMLDOM_Node *pCurNode = pStartNode;
	pCurNode->AddRef();

	while (true)
	{
		if (QueryItem.findType == QUERYITEM::_DOUBLE)
		{
			qrResult = FindNext(pCurNode, QueryItem);
		}
		else
		{
			XMLDOM_NodeList *attrList;
			if (Compare(QueryItem, pCurNode, &attrList, vecIndex))
			{
				if (attrList) 
				{
					qrResult = AddNodesToFoundList(attrList);
					
					attrList->Release();

					break;
				}
				else
				{
					qrResult = ExecuteQuery(QueryItem.sNextQuery, pCurNode);
				} 

			}
			else if (attrList)
			{
				attrList->Release();
			}
			
			if (QueryItem.nodeType == QUERYITEM::NODE_ATTRIBUTE) 
			{
				break;
			}
		}
		
		if (qrResult == _ENDING)
		{
			break;
		}

		XMLDOM_Node *pNextNode;
		if (!MoveNext(pCurNode, pNextNode))
		{
			break;
		}

		pCurNode->Release();
		pCurNode = pNextNode;
	} 

	pCurNode->Release();

	return qrResult;
}


bool CXPath::ParseQuery(string& asQuery, list<QUERYITEM> &listQueryItem)
{
	int iBrCounter = 0;
	string::size_type iPosSl = string::npos;
	int iCurSlashPrior = 1000;
	vector<Union>vUnion;
	int iCurUnionPrior = 1000;
	enum DELIMITER {_SLASH, _CLOSE, _NONE};
	
	string::size_type pos = 0;
	string::size_type startpos = 0;
	string::size_type res;
	
	TrimStr(asQuery);
	DELIMITER lastEl = _NONE;
	RemoveOrphanBrackets(asQuery);

	string::size_type len = asQuery.size();
	if (!len) 
	{
		return false;
	}
	
	QUERYITEM::FINDTYPE SlType = QUERYITEM::_UNDEF;
	bool bFirstLet = false;

	while (pos < len)
	{
		switch(((const string &)asQuery)[pos]) 
		{
			//  [4/18/2008 Andriy.Vitkovskyy]
			//  Found need square bracket ']' which correspond to first square bracket '['.
			//	If present situation [...[...]..] then 
			//	if found second '[' then need found two ']'. 
			//	Skip square bracket in text value.			
		case '[':
		case '\"':
			{	
				bool bRepeat = false;
				int nNumSqBr = 0;
				string::size_type nPosSqBr = pos;	
		lRepeat:
				pos = asQuery.find( ']', nPosSqBr);			
			if ( pos == string::npos ) 
			{
				return false;
			}//if
				else
				{
					while (bRepeat || !nNumSqBr)
					{			
						bRepeat = false;
						while (pos >= nPosSqBr)
						{								
							switch(((const string &)asQuery)[nPosSqBr])
							{
							case '[':
								nNumSqBr++;
								break;
							case ']':
								nNumSqBr--;
								break;
							case '\"':
							case '\'':
								NextQuote(asQuery, nPosSqBr);
								nPosSqBr++;
								if(pos <= nPosSqBr)
								{
									bRepeat = true;
									goto lRepeat;							
								}
							}
							nPosSqBr++;				

						}//end while (pos >= nPosSqBr)
						if((pos <= nPosSqBr) && (nNumSqBr))
						{
							bRepeat = true;
							goto lRepeat;
						}
						if (!nNumSqBr || pos == (string::npos) || (pos > len))
						{
			break;
						}

					}//end while (bRepeat || !nNumSqBr)
				}

				break;
			}
		case '(':
			++iBrCounter;
			break;

		case ')':
			--iBrCounter;
			break;

		case '|':
			if (iBrCounter < iCurUnionPrior) 
				iCurUnionPrior = iBrCounter;
			vUnion.push_back(Union((int)pos, iCurUnionPrior));
			break;

		case '/':
			if (!bFirstLet)
			{
				lastEl = _SLASH;
				if (pos+1<len && ((const string &)asQuery)[pos+1]=='/')
				{
					SlType = QUERYITEM::_DOUBLE;
					pos += 2;
					startpos = pos;
					continue;
				}

				++pos;
				startpos = pos;
				SlType = QUERYITEM::_SINGLE;
				continue;
			}

			if (iBrCounter < iCurSlashPrior)
			{
				iCurSlashPrior = iBrCounter;
				iPosSl = pos;
			}

			break;

		default:
			bFirstLet = true;
			break;
		}

		++pos;
	}

	int minPrior = 1000;
	if (iCurUnionPrior > iCurSlashPrior)
	{
		//find min prior and del all element woth pos > iPosSl
		vector<Union>::iterator it=vUnion.begin();
		if (iPosSl != string::npos)
		{
			for (; it!=vUnion.end() && it->pos < (int)iPosSl; ++it)
			{
				if (minPrior > it->prior)
				{
					minPrior = it->prior;
				}
			}
		}
		vUnion.erase(it, vUnion.end());
	}
	else
	{
		vector<Union>::iterator it;
		for (it=vUnion.begin(); it!=vUnion.end(); ++it)
		{
			if (minPrior > it->prior)
			{
				minPrior = it->prior;
			}
		}
	}

	if (vUnion.empty()) 
	{
		//single query
		QUERYITEM item;

		if (iPosSl == string::npos)
		{
			iPosSl = asQuery.size();
		}

		item.sNextQuery = asQuery.substr(iPosSl, asQuery.size() - iPosSl);
		item.findType = SlType;
		
		bool bName = false;
		string::size_type i=startpos;

		while (i<iPosSl)
		{
			if (asQuery[i] == '[')
			{
				if (!bName)
				{
					item.AddName(asQuery.substr(startpos, i - startpos));
					bName = true;
				}

				//res = asQuery.find_last_of("]", iPosSl);
				//res = asQuery.find("]", i);
				NextSquareBr(asQuery, res = i);				
				if (res == string::npos)
					return false;

				item.Condition.push_back(asQuery.substr(i + 1, res-i-1));
				startpos = res + 1;
				i = startpos;

				continue;
			}

			++i;
		}

		if (!bName)
		{
			item.AddName(asQuery.substr(startpos, i - startpos));
		}

		listQueryItem.push_back(item);
	}
	else
	{
		//multy query
		vector<Union>::iterator it;
//		int start = startpos;
		int start = 0;

		for (it=vUnion.begin(); it!=vUnion.end(); ++it)
		{
			if (it->prior == minPrior)
			{
				QUERYITEM item;

				item.ItemClear();
				item.sNextQuery = asQuery.substr(start, it->pos - start);
				if (iCurUnionPrior > iCurSlashPrior && iPosSl != string::npos)
				{
					item.sNextQuery += asQuery.substr(iPosSl, asQuery.size() - iPosSl);
				}
				TrimStr(item.sNextQuery);
				item.findType = SlType;
				listQueryItem.push_back(item);

				start = it->pos + 1;
			}
		}

		QUERYITEM item;
		item.ItemClear();
		item.sNextQuery = asQuery.substr(start, asQuery.size() - start);

/*		if (iCurUnionPrior > iCurSlashPrior)
		{
			item.sNextQuery += asQuery.substr(iPosSl, asQuery.size() - iPosSl);
		}*/

		TrimStr(item.sNextQuery);
		listQueryItem.push_back(item);
	}

	return true;
}

QUERYRESULT CXPath::FindNext(XMLDOM_Node *node, QUERYITEM& QueryItem)
{
	QUERYRESULT qrResult = _FALSE;

	vector<int> vecIndex;

	XMLDOM_Node *endNode = node;

	XMLDOM_Node *tmpNode = node; 
	tmpNode->AddRef();

	bool Enter = (QUERYITEM::DOMNodeTypeEX)node->GetNodeType() == QUERYITEM::NODE_DOCUMENT && QueryItem.nodeType == QUERYITEM::NODE_ATTRIBUTE;

	while (true)
	{
		bool Ending = false;

		XMLDOM_NodeList *attrList = NULL;
		while (Enter || !Compare(QueryItem, tmpNode, &attrList, vecIndex))
		{
			if (attrList)
			{
				attrList->Release();
				attrList = NULL;
			}

			Enter = false;

			XMLDOM_Node *otherNode;
			if (!MoveSubTree(tmpNode, otherNode, endNode))
			{
				Ending = true;
				break;
			}

			tmpNode->Release();
			tmpNode = otherNode;
		}

		if (Ending) 
		{
			assert(!attrList);
			break;
		}

		if (attrList) 
		{
			qrResult = AddNodesToFoundList(attrList);

			attrList->Release();
			
		}
		else
		{
			qrResult = ExecuteQuery(QueryItem.sNextQuery, tmpNode);
		} 

		if (qrResult == _ENDING)
		{
			break;
		}

		Enter = true;
	}

	tmpNode->Release();

	return qrResult;
}


bool CXPath::DefineStart(XMLDOM_Node *node, XMLDOM_Node *&startNode, bool &bSkipMovingDown, QUERYITEM &QueryItem)
{
	bool bResult = false;
	startNode = NULL;
	bSkipMovingDown = false;
	
	do
	{
		if (!node)
		{
			node = m_StartNode;
			
			if (QueryItem.findType == QUERYITEM::_ANCESTOR) 
			{
				if (node->IsElementOrDescendant())
				{
					bResult = static_cast<XMLDOM_Element *>(node)->SelectSingleNode(QueryItem.name, startNode);
				}

				break;
			}
			else if (QueryItem.name.compare(".") == 0 || QueryItem.name.compare("..") == 0)
			{
			} 
			else if (QueryItem.findType == QUERYITEM::_UNDEF)
			{
				QueryItem.findType = QUERYITEM::_SINGLE;
			}
			else
			{
				GetUltimateNodeParent(node, startNode);
				
				bSkipMovingDown = true;
				bResult = true;
				break;
			}
		}

		node->AddRef();
		startNode = node;
		bResult = true;
	}
	while (false);
	
	return bResult;
}

void CXPath::GetUltimateNodeParent(XMLDOM_Node *node, XMLDOM_Node *&parentNode)
{
	if (node->GetNodeType() != NODE_DOCUMENT)
	{
		GetUltimateNodeParent_Ascend(node, parentNode);
	}
	else
	{
		GetUltimateNodeParent_DescendRoot(static_cast<XMLDOM_Document *>(node), parentNode);
	}
}

void CXPath::GetUltimateNodeParent_Ascend(XMLDOM_Node *node, XMLDOM_Node *&parentNode)
{
	XMLDOM_Node *pnCurNode = node;
	pnCurNode->AddRef();

	while (true)
	{
		XMLDOM_Element *peCurParent;
		if (!pnCurNode->GetParentNode(peCurParent))
		{
//			pnCurNode->AddRef(); -- optimization
			parentNode = pnCurNode;
			break;
		}

		if (peCurParent->GetNodeType() == NODE_DOCUMENT)
		{
			peCurParent->Release();

//			pnCurNode->AddRef(); -- optimization
			parentNode = pnCurNode;
			break;
		}

		pnCurNode->Release();
		pnCurNode = peCurParent;
	}

//	pnCurNode->Release(); -- optimization
}

void CXPath::GetUltimateNodeParent_DescendRoot(XMLDOM_Document *document, XMLDOM_Node *&parentNode)
{
	XMLDOM_Element *peDocumentElement;
	document->GetDocumentElement(peDocumentElement);

	parentNode = peDocumentElement;
}

QUERYRESULT CXPath::AddNodesToFoundList(XMLDOM_NodeList *nodeList)
{
	QUERYRESULT qrResult = _FALSE;

	XMLDOM_NodeList::size_type nItemCount = nodeList->GetLength();
	for (XMLDOM_NodeList::size_type nItemIndex = 0; nItemIndex != nItemCount; ++nItemIndex)
	{
		XMLDOM_Node *node;
		nodeList->GetItem(nItemIndex, node);
		
		m_FoundList->AddItem(node);
		
		node->Release();
		
		if (CheckEnding() == _ENDING)
		{
			qrResult = _ENDING;
			break;
		}
	}

	return qrResult;
}

QUERYRESULT CXPath::CheckEnding()
{
	return (m_Number == SELECT_SINGLE && 1 <= (int)m_FoundList->GetLength()) ? _ENDING : _FALSE;
}


bool CXPath::Compare(QUERYITEM &QueryItem, XMLDOM_Node *node, XMLDOM_NodeList **pAttrList, vector<int> &vecIndex, int iQuerySize)
{
	XMLDOM_NodeList *attrList = NULL;
	if (iQuerySize == -1)
	{
		iQuerySize = (int)QueryItem.Condition.size();
	}

	bool bResult;

	DOMNodeType ntNodeType = node->GetNodeType();

	switch (QueryItem.nodeType) 
	{
	case QUERYITEM::NODE_ELEMENT:
		bResult = CompareElement(QueryItem, node, vecIndex, iQuerySize);
		break;

	case QUERYITEM::NODE_ATTRIBUTE:
		bResult = CompareAttribute(QueryItem, node, attrList, iQuerySize);
		break;

	case QUERYITEM::NODE_CDATA_SECTION:
		bResult = ntNodeType == NODE_CDATA_SECTION;
		break;

	case QUERYITEM::NODE_COMMENT:
		bResult = ntNodeType == NODE_COMMENT;
		break;

	case QUERYITEM::NODE_TEXT:
		bResult = ntNodeType == NODE_TEXT;
		break;

	case QUERYITEM::NODE_TEXT_CDATA:
		bResult = ntNodeType == NODE_TEXT || ntNodeType == NODE_CDATA_SECTION;
		break;

	case QUERYITEM::NODE_NODE:
		bResult = true;
		break;

	default:
		bResult = false;
	}

	if (pAttrList)
	{
		*pAttrList = attrList;
	}
	else if (attrList)
	{
		attrList->Release();
	}

	return bResult;
}

bool CXPath::CompareElement(QUERYITEM &QueryItem, XMLDOM_Node* node, vector<int> &vecIndex, int iQuerySize)
{
	bool bResult = false;
	
	do
	{
		if (node->GetNodeType() != NODE_ELEMENT)
		{
			break;
		}
	
		const string &sNodeName = static_cast<XMLDOM_Element *>(node)->GetTagName();
		int iNodeNameComparisonResult =
			CompareString(sNodeName.c_str(), QueryItem.name.c_str(), !m_bCaseInsensitive);
		if (iNodeNameComparisonResult != 0 && QueryItem.name.compare("*") != 0)
		{
			break;
		}

		//execute filters
		unsigned int filterNo = 0;
		for (; (int)filterNo < iQuerySize; ++filterNo)
		{
			if (vecIndex.size() <= filterNo)
				vecIndex.push_back(0);
			else
				++(vecIndex[filterNo]);

			if (!ExecuteFilter(node, QueryItem, filterNo, vecIndex[filterNo])) 
			{
				break;
			}
		}
		if ((int)filterNo < iQuerySize) 
		{
			break;
		}

		bResult = true;
	}
	while (false);
	
	return bResult;
}

bool CXPath::CompareAttribute(QUERYITEM &QueryItem, XMLDOM_Node *node, XMLDOM_NodeList *&attrList, int iQuerySize)
{
	bool bResult = false;

	if (node->GetNodeType() == NODE_ELEMENT)
	{
		if (!QueryItem.name.compare("*"))
		{
			bResult = CompareAttribute_asterisk(QueryItem, node, attrList, iQuerySize);
		}
		else
		{
			bResult = CompareAttribute_other(QueryItem, node, attrList, iQuerySize);
		}
	}

	return bResult;
}

bool CXPath::CompareAttribute_asterisk(QUERYITEM &QueryItem, XMLDOM_Node *node, XMLDOM_NodeList *&attrList, int iQuerySize)
{
	static_cast<XMLDOM_Element *>(node)->GetAttributes(attrList);
	
	int filterNo = 0;
	for (; filterNo < iQuerySize; ++filterNo)
	{
		int i = 0;
		XMLDOM_NodeList::size_type nItemCount = attrList->GetLength();
		for (XMLDOM_NodeList::size_type nItemIndex = 0; nItemIndex != nItemCount;)
		{
			XMLDOM_Node *pAttr;
			attrList->GetItem(nItemIndex, pAttr);

			if (!ExecuteFilter(pAttr, QueryItem, filterNo, i, attrList))
			{
				attrList->RemoveItem(nItemIndex);
				--nItemCount;
			}
			else
			{
				++nItemIndex;
			}
			
			pAttr->Release();
			++i;
		}
	}
	
	bool bResult = attrList->GetLength() != 0;
	return bResult;
}

bool CXPath::CompareAttribute_other(QUERYITEM &QueryItem, XMLDOM_Node *node, XMLDOM_NodeList *&attrList, int iQuerySize)
{
	bool bResult = false;

	XMLDOM_Attr *attrNode; 
	if (static_cast<XMLDOM_Element *>(node)->GetAttributeNode(QueryItem.name, attrNode))
	{
		//execute filters
		int filterNo = 0;
		for (; filterNo < iQuerySize; ++filterNo)
		{
			if (!ExecuteFilter(attrNode, QueryItem, filterNo, 0))
			{
				break;
			}
		}
		
		if (filterNo == iQuerySize)
		{
			attrList = new XMLDOM_NodeList();
			attrList->AddItem(attrNode);
			
			bResult = true;
		}

		attrNode->Release();
	}

	return bResult;
}


bool CXPath::ExecuteFilter(XMLDOM_Node *node, QUERYITEM& QueryItem, int FilterNo, int iSuccIndex, XMLDOM_NodeList *pAttrList)
{
	bool bResult;
	
	const string &str = QueryItem.Condition[FilterNo];

	if (str.compare("end()") == 0)
	{
		bResult = ExecuteFilter_end(node, QueryItem, FilterNo, pAttrList);
	}
	else
	{
		bResult = ExecuteFilter_standard(str, node, iSuccIndex);
	}

	return bResult;
}

bool CXPath::ExecuteFilter_end(XMLDOM_Node *node, QUERYITEM& QueryItem, int FilterNo, XMLDOM_NodeList *pAttrList)
{
	bool bResult;
	
	if (pAttrList) 
	{
		bResult = ExecuteFilter_end_AttrList(node, pAttrList);
	}
	else
	{
		bResult = ExecuteFilter_end_other(node, QueryItem, FilterNo);
	}
	
	return bResult;
}

bool CXPath::ExecuteFilter_end_AttrList(XMLDOM_Node *node, XMLDOM_NodeList *pAttrList)
{
	bool bResult = false;
	
	do
	{
		XMLDOM_NodeList::size_type nAttrCount = pAttrList->GetLength();
		if (nAttrCount == 0)
		{
			break;
		}
		
		XMLDOM_Node *pnLastAttr;
		pAttrList->GetItem(nAttrCount - 1, pnLastAttr);
		
		bResult = node == pnLastAttr;
		
		pnLastAttr->Release();
	
		bResult = true;
	}
	while (false);
	
	return bResult;
}

bool CXPath::ExecuteFilter_end_other(XMLDOM_Node *node, QUERYITEM& QueryItem, int FilterNo)
{
	bool bResult = true;

	XMLDOM_Node *pCurNode = node, *pNextNode;
	pCurNode->AddRef();
	
	while (MoveNext(pCurNode, pNextNode))
	{
		pCurNode->Release();
		pCurNode = pNextNode;
		
		//compare
		vector<int> vecIndex;
		if (Compare(QueryItem, pCurNode, NULL, vecIndex, FilterNo))
		{
			bResult = false;
			break;
		}
	}
	
	pCurNode->Release();

	return bResult;
}

bool CXPath::ExecuteFilter_standard(const string &str, XMLDOM_Node *node, int iSuccIndex)
{
	CFilterExpr Filter(node, iSuccIndex);

	Filter.Parse(str);

	bool bResult = Filter.Evaluate();
	return bResult;
}



bool CXPath::MoveNext(XMLDOM_Node *node, XMLDOM_Node *&nextNode)
{
	return node->GetNextSiblingNode(nextNode);
}

bool CXPath::MovePrev(XMLDOM_Node *node, XMLDOM_Node *&prevNode)
{
	return node->GetPreviousSiblingNode(prevNode);
}

bool CXPath::MoveUp(XMLDOM_Node *node, XMLDOM_Node *&upperNode)
{
	XMLDOM_Element *upperElement;
	bool bResult = node->GetParentNode(upperElement);
	
	upperNode = upperElement;
	return bResult;
}

bool CXPath::MoveDown(XMLDOM_Node *node, XMLDOM_Node *&lowerNode, QUERYITEM& QueryItem)
{
	if (QueryItem.nodeType == QUERYITEM::NODE_ATTRIBUTE)
	{
		node->AddRef();
		lowerNode = node;
		return true;
	}

	return MoveDown(node, lowerNode);
}

bool CXPath::MoveDown(XMLDOM_Node *node, XMLDOM_Node *&lowerNode)
{
	lowerNode = NULL;

	if (!node->IsElementOrDescendant())
	{
		return false;
	}
	
	return static_cast<XMLDOM_Element *>(node)->GetFirstChildNode(lowerNode);
}

bool CXPath::MoveSubTree(XMLDOM_Node *node, XMLDOM_Node *&otherNode, XMLDOM_Node *endNode)
{
	bool bResult = false;

	if (MoveDown(node, otherNode))
	{
		bResult = true;
	}
	else
	{
		XMLDOM_Node *tmpNode = node;
		tmpNode->AddRef();
		
		while (true) 
		{
			if (endNode == tmpNode) 
			{
				break;
			}
			
			if (MoveNext(tmpNode, otherNode))
			{
				bResult = true;
				break;
			}
			else
			{
				if (!MoveUp(tmpNode, otherNode))
				{
					break;
				}
				
				tmpNode->Release();
				tmpNode = otherNode;
			}
		}

		tmpNode->Release();
	}

	return bResult;
}
