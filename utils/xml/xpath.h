#if !defined(AFX_XPATH_H__4BD5637B_0244_4FC1_B252_FFB93847D8AC__INCLUDED_)
#define AFX_XPATH_H__4BD5637B_0244_4FC1_B252_FFB93847D8AC__INCLUDED_

#include <assert.h>
#include "xmldomintf.h"

#if _MSC_VER > 1000
#pragma once
#endif // _MSC_VER > 1000

using namespace std;

enum QUERYRESULT {_FALSE, _ENDING}; 

struct QUERYITEM
{
	enum DOMNodeTypeEX {
		NODE_INVALID = ::NODE_INVALID,
		NODE_ELEMENT = ::NODE_ELEMENT,
		NODE_ATTRIBUTE = ::NODE_ATTRIBUTE,
		NODE_TEXT = ::NODE_TEXT,
		NODE_CDATA_SECTION = ::NODE_CDATA_SECTION,
		NODE_COMMENT = ::NODE_COMMENT,
		NODE_DOCUMENT = ::NODE_DOCUMENT,

		NODE_TEXT_CDATA = ::NODE__MAX,
		NODE_NODE
	};

	enum FINDTYPE {_SINGLE, _DOUBLE, _ANCESTOR, _UNDEF} findType;
	string name;
	DOMNodeTypeEX nodeType;
	vector<string> Condition;
	string sNextQuery;

	void ItemClear();
	void AddName(const string& asName);
};

class CXPath
{
	struct Union
	{
		int pos;
		int prior;
		Union(int p, int pr): pos(p), prior(pr){};
	};
	
public:
	enum
	{
		SELECT_ALL,
		SELECT_SINGLE	// Oder-styled comma removed, for 'CC' UNIX compiler only
	};

	//CXPath();
	CXPath(XMLDOM_Node *pNode, const string& asPattern, XMLDOM_NodeList *foundList, int iNumber=SELECT_ALL, bool bCaseInsensitive = false);
	virtual ~CXPath();
	void Search();	
	
private:
	QUERYRESULT ExecuteQuery(string& sPattern, XMLDOM_Node* pStartNode);
	QUERYRESULT ExecuteSingleQueryItem(XMLDOM_Node *pStartNode, QUERYITEM &QueryItem, vector<int> &vecIndex);
	QUERYRESULT ExecuteQueryItemForLevelNodes(XMLDOM_Node *pStartNode, QUERYITEM &QueryItem, vector<int> &vecIndex);
	
	bool ParseQuery(string& sQuery, list<QUERYITEM>& QueryItem);
	QUERYRESULT FindNext(XMLDOM_Node *node, QUERYITEM& QueryItem);

	bool DefineStart(XMLDOM_Node *node, XMLDOM_Node *&startNode, bool &bSkipMovingDown, QUERYITEM &QueryItem);
	static void GetUltimateNodeParent(XMLDOM_Node *node, XMLDOM_Node *&parentNode);
	static void GetUltimateNodeParent_Ascend(XMLDOM_Node *node, XMLDOM_Node *&parentNode);
	static void GetUltimateNodeParent_DescendRoot(XMLDOM_Document *document, XMLDOM_Node *&parentNode);
	QUERYRESULT AddNodesToFoundList(XMLDOM_NodeList *nodeList);
	QUERYRESULT CheckEnding();
	
	bool Compare(QUERYITEM &QueryItem, XMLDOM_Node* node, XMLDOM_NodeList **pAttrList, vector<int> &vecIndex, int iQuerySize = -1);
	bool CompareElement(QUERYITEM &QueryItem, XMLDOM_Node* node, vector<int> &vecIndex, int iQuerySize);
	bool CompareAttribute(QUERYITEM &QueryItem, XMLDOM_Node *node, XMLDOM_NodeList *&attrList, int iQuerySize);
	bool CompareAttribute_asterisk(QUERYITEM &QueryItem, XMLDOM_Node *node, XMLDOM_NodeList *&attrList, int iQuerySize);
	bool CompareAttribute_other(QUERYITEM &QueryItem, XMLDOM_Node *node, XMLDOM_NodeList *&attrList, int iQuerySize);
	
	bool ExecuteFilter(XMLDOM_Node*, QUERYITEM& QueryItem, int FilterNo, int iSuccNum, XMLDOM_NodeList *pAttrList = NULL);
	bool ExecuteFilter_end(XMLDOM_Node *node, QUERYITEM& QueryItem, int FilterNo, XMLDOM_NodeList *pAttrList);
	bool ExecuteFilter_end_AttrList(XMLDOM_Node *node, XMLDOM_NodeList *pAttrList);
	bool ExecuteFilter_end_other(XMLDOM_Node *node, QUERYITEM& QueryItem, int FilterNo);
	bool ExecuteFilter_standard(const string &str, XMLDOM_Node *node, int iSuccIndex);

	bool MoveNext(XMLDOM_Node *node, XMLDOM_Node *&nextNode);
	bool MovePrev(XMLDOM_Node *node, XMLDOM_Node *&prevNode);
	bool MoveDown(XMLDOM_Node *node, XMLDOM_Node *&lowerNode, QUERYITEM& QueryItem);
	bool MoveUp(XMLDOM_Node *node, XMLDOM_Node *&upperNode);
	bool MoveDown(XMLDOM_Node *node, XMLDOM_Node *&lowerNode);
	bool MoveSubTree(XMLDOM_Node *node, XMLDOM_Node *&otherNode, XMLDOM_Node *endNode);
	
private:
	XMLDOM_Node *m_StartNode;
	XMLDOM_NodeList *m_FoundList;
	string m_QueryString;
	int m_Number;
	bool m_bCaseInsensitive;
};

#endif // !defined(AFX_XPATH_H__4BD5637B_0244_4FC1_B252_FFB93847D8AC__INCLUDED_)
