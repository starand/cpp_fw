// FilterExpr.h: interface for the CFilterExpr class.
//
//////////////////////////////////////////////////////////////////////

#if !defined(AFX_FILTEREXPR_H__C0DE87BD_7C04_4C4D_9B58_74D134942B9A__INCLUDED_)
#define AFX_FILTEREXPR_H__C0DE87BD_7C04_4C4D_9B58_74D134942B9A__INCLUDED_

#if _MSC_VER > 1000
#pragma once
#endif // _MSC_VER > 1000

#include "xmldomintf.h"
using namespace std;

// Forward declarations:
struct _VARIANT;

enum EXPRTOKEN {
	_OPENBR = 10, _CLOSEBR, 
	_ALL, _ANY,
	_EQ, _IEQ, _LT, _ILT, _GT, _IGT, _GE, _IGE, _LE, _ILE,	_NE, _INE,					
	_NOT,
	_AND, 
	_OR, 
	_LITERAL, _NODE, _NUMERIC, _DATE,
	_NONE}; 

enum METHOD {M_VALUE, M_NODENAME, M_NODETYPE, M_NUMBER, M_END, M_INDEX };

struct Literal
{
	Literal(): type(_NONE) {}
	Literal(const string& str, EXPRTOKEN _type): name(str), type(_type) {}
	string name;
	EXPRTOKEN type;
	vector<METHOD> listMethod;
};

typedef list<int>::iterator LIST_IT;

class CFilterExpr  
{
public:
	CFilterExpr(XMLDOM_Node* node, int&iSuccIndex);
	~CFilterExpr();
	void Parse(const string& asExpr);
	bool Evaluate();
//	void Write();

protected:
	void Pop();
	EXPRTOKEN GetPop();
	void Push(EXPRTOKEN, Literal&);
	LIST_IT GetNextOperator();
	bool BoolEvaluate(int operand);
	bool StringEvaluate(LIST_IT itOper, int l_operand, int r_operand);
	bool EvalStrOperator(LIST_IT itOper, _VARIANT &lVar, _VARIANT &rVar);
	bool EvalNodeOperators(LIST_IT itOper, int l_operand, _VARIANT &lVar, _VARIANT &rVar);
	bool EvaluateExpr(LIST_IT itOper, LIST_IT itBegin);
	void InsertResult(LIST_IT itOper, int iRes);
	void ExecuteMethods(XMLDOM_Node *pNode, Literal& litToken, _VARIANT &Var);

	EXPRTOKEN NextToken(const string& asExpr, string::size_type &aiPos,  Literal& litToken);		
	list<EXPRTOKEN> m_Stack;
	vector<Literal> m_LitTable;
	list<int> m_Parsed;
	XMLDOM_Node *m_Node;
	EXPRTOKEN m_SetOperator;
	int m_SuccIndex;
};

bool NextSquareBr(const string& asExpr, string::size_type &aiPos);
bool NextQuote(const string& asExpr, string::size_type &aiPos);

#endif // !defined(AFX_FILTEREXPR_H__C0DE87BD_7C04_4C4D_9B58_74D134942B9A__INCLUDED_)
