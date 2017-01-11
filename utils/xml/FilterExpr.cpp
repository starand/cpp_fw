// FilterExpr.cpp: implementation of the CFilterExpr class.
//
//////////////////////////////////////////////////////////////////////

#include "StdAfx.h"
#include <time.h>
#include "strutils.h"
#include "FilterExpr.h"

#include <time.h>

#if defined(LINUX) || defined(UNIX)
#include <ctype.h>
#endif

#include "strutils.h"

using namespace StrUtils;

struct _VARIANT
{
	_VARIANT(): iVal(0), varType(_UNDEF) {};
	_VARIANT(void* val, int type, XMLDOM_Node *pNode = NULL);
	
	string ExtractDateFromNode(XMLDOM_Node *pNode, const string &sPattern);
	
	void ToStr();
	void ToInt();
	void ToDate();
	void Clear();
	
	string sVal;
	int iVal;
	struct tm dateVal;  
	
	enum {_UNDEF = -1, _STR = 0, _INT = 1, _DATE = 2} varType;
};

const int TOKEN_Priority[] = 
{
	0, 0, 
	4, 4, 
	6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,					
	8,
	9, 
	10, 
	-1, -1, 
	-1
};

void TrimAll(string& sTemp, char chLeft=' ', char chRight=' ')
{
	string::size_type iStart = 0, iEnd = sTemp.size();
	
	while (iStart < iEnd)
	{
		char ch = ((const string &)sTemp)[iStart];
		if (((unsigned char)ch <= (unsigned char)' ') || (ch == chLeft))
		{
			++iStart;
		}
		else
		{
			do
			{
				ch = ((const string &)sTemp)[--iEnd];
			} 
			while (((unsigned char)ch <= (unsigned char)' ') || (ch == chRight));
			
			++iEnd;
			break;
		}
	}
	
	if(iEnd != sTemp.size())
		sTemp.erase(iEnd);
	
	if(iStart)
		sTemp.erase(0, iStart);
}

inline void TrimStr(string& sTemp)
{
	TrimAll(sTemp);
}

_VARIANT::_VARIANT(void* val, int type, XMLDOM_Node *pNode):
	varType(_UNDEF)
{
	switch(type) 
	{
	case _INT:
		varType = _INT;
		iVal = *(int*)val; 
		
		break;

	case _STR:
		varType = _STR;
		sVal = *(string*)val; 
		
		break;

	case _DATE:
		string s = *(string*)val;
		TrimStr(s);
		
		if(!s.empty() && (((const string &)s)[0] == '\"' || ((const string &)s)[0] == '\''))
		{
			s.erase(0,1);
			if(!s.empty() && (((const string &)s)[s.size() - 1] == '\"' || ((const string &)s)[s.size() - 1] == '\'')) 
			{
				s.erase(s.size() - 1,1);
			}
		}
		else
		{
			s = ExtractDateFromNode(pNode, s);
		}

		int iMonth, iYear, iDay;
		if (sscanf(s.c_str(), "%d-%d-%d", &iMonth, &iDay, &iYear) == 3)
		{
			dateVal.tm_mon = (int)iMonth;
			dateVal.tm_mday = (int)iDay;
			dateVal.tm_year = (int)(iYear - 1900);
			varType = _DATE;
		}

		break;
	}
}

string _VARIANT::ExtractDateFromNode(XMLDOM_Node *pNode, const string &sPattern)
{
	string sResult;

	if (pNode && pNode->GetNodeType() == NODE_ELEMENT)
	{
		XMLDOM_Node *pResultNode;
		if (static_cast<XMLDOM_Element *>(pNode)->SelectSingleNode(sPattern, pResultNode))
		{
			pResultNode->GetText(sResult); 
			
			pResultNode->Release();
		}
	}

	return sResult;
}

void _VARIANT::Clear()
{
	sVal.erase();
	varType = _STR;
}

void _VARIANT::ToStr()
{
	if(varType == _STR)
		return;
	
	char buffer[32] = { 0 };
	sprintf(buffer, "%d", iVal);
	sVal = string(buffer);
	varType = _STR;
}

void _VARIANT::ToInt()
{
	if(varType == _INT)
		return;
	if (varType == _STR)
	{
		iVal = atoi(sVal.c_str());
		sVal.erase();
		varType = _INT;
	}
}

void _VARIANT::ToDate()
{
	try
	{
		if(varType == _DATE)
			return;
		if (varType == _STR)
		{
			if(sVal.empty()) return;
			string s= sVal;
			TrimStr(s);
			if(!s.empty() && (((const string &)s)[0] == '\"' || ((const string &)s)[0] == '\''))
			{
				s.erase(0,1);
			}
			if(!s.empty() && (((const string &)s)[s.size() - 1] == '\"' || ((const string &)s)[s.size() - 1] == '\''))
			{
				s.erase(s.size() - 1,1);
			}
			int iMonth, iYear, iDay;
			if (sscanf(s.c_str(), "%d-%d-%d", &iMonth, &iDay, &iYear) == 3)
			{
				dateVal.tm_mon = (int)iMonth;
				dateVal.tm_mday = (int)iDay;
				dateVal.tm_year = (int)(iYear - 1900);
				varType = _DATE;
			}
		}
	}
	catch (...) 
	{
		varType = _UNDEF;
	}
}

//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////

CFilterExpr::CFilterExpr(XMLDOM_Node *node, int&iSuccIndex): 
	m_Node(node), 
	m_SuccIndex(iSuccIndex),
	m_SetOperator(_NONE)
{
}

CFilterExpr::~CFilterExpr()
{
}

const int TK_LIT = 100;
//////////////////////////////////////////////////////////////////////////
// CFilterExpr
//////////////////////////////////////////////////////////////////////////

inline bool IsSpace(const string &str, const string::size_type pos)
{
	char cSym = str[pos];
	return ((cSym == ' ') || (cSym == '\t') || (cSym == '\r') || (cSym == '\n'));
}

void SkipWhiteSpaces(const string&asExpr, string::size_type& aiPos)
{
	for(; aiPos < asExpr.size() && IsSpace(asExpr, aiPos); aiPos++);
}

void CFilterExpr::Pop()
{
	EXPRTOKEN token;
	token = *m_Stack.rbegin();
	m_Stack.pop_back();
	if(token != _OPENBR)
	{
		m_Parsed.push_back(token);
	}
}

EXPRTOKEN CFilterExpr::GetPop()
{
	if(m_Stack.empty())
		return _NONE;
	return *m_Stack.rbegin();
}

void CFilterExpr::Push(EXPRTOKEN token, Literal& litToken)
{
	switch(token) 
	{
	case _CLOSEBR:
		while(GetPop() != _OPENBR)
		{
			Pop();
		}//while
		Pop();
		break;
	case _OPENBR:
		m_Stack.push_back(token);
		break;
	case _DATE:
	case _NUMERIC:
	case _LITERAL:
	case _NODE:
		{
			litToken.type = token;
			m_Parsed.push_back(TK_LIT + (int)m_LitTable.size());
			m_LitTable.push_back(litToken);
		}
		break;
	default :
		while(TOKEN_Priority[GetPop() - 10] <= TOKEN_Priority[token - 10] && GetPop() != _OPENBR )
			Pop();
		m_Stack.push_back(token);
		break;
	}
}
 
EXPRTOKEN CFilterExpr::NextToken(const string& asExpr, string::size_type &aiPos, Literal& litToken)
{
	litToken.name.erase();
	litToken.listMethod.clear();
	
	SkipWhiteSpaces(asExpr, aiPos);
	if(aiPos >= asExpr.length())
	{
		return _NONE;
	}
	switch(asExpr[aiPos])
	{
	case '(' : 
		aiPos ++;
		return _OPENBR;
	case ')' : 
		aiPos ++;
		return _CLOSEBR;
	case '$' :
		{
			string::size_type res = asExpr.find("$", aiPos + 1);
			string oper = asExpr.substr(aiPos + 1, res - aiPos - 1);
			aiPos = res + 1;
			if (CompareStr(oper.c_str(), "ieq"))  return _IEQ;
			if (CompareStr(oper.c_str(), "eq"))  return _EQ;
			if (CompareStr(oper.c_str(), "lt"))  return _LT;
			if (CompareStr(oper.c_str(), "ilt"))  return _ILT;
			if (CompareStr(oper.c_str(), "gt"))  return _GT;
			if (CompareStr(oper.c_str(), "igt"))  return _IGT;
			if (CompareStr(oper.c_str(), "ge"))  return _GE;
			if (CompareStr(oper.c_str(), "ige"))  return _IGE;
			if (CompareStr(oper.c_str(), "le"))  return _LE;
			if (CompareStr(oper.c_str(), "ile"))  return _ILE;
			if (CompareStr(oper.c_str(), "ne"))  return _NE;
			if (CompareStr(oper.c_str(), "ine"))  return _INE;
		
			if (CompareStr(oper.c_str(), "and"))  return _AND;
			if (CompareStr(oper.c_str(), "or"))  return _OR;
			if (CompareStr(oper.c_str(), "not"))  return _NOT;
			
			if (CompareStr(oper.c_str(), "all"))  return _ALL;
			if (CompareStr(oper.c_str(), "any"))  return _ANY;
			
			break;
		}
	case '&' :
		if(asExpr[++aiPos] == '&')
		{
			aiPos++;
			return _AND;
		}
		return _NONE;
	case '|' :
		if(asExpr[++aiPos] == '|')
		{
			aiPos++;
			return _OR;
		}
		return _NONE;
	case '=' :
		aiPos ++;
		return _EQ;
	case '!' :
		if (asExpr[aiPos + 1] != '=') 
			return _NONE;
		aiPos += 2;
		return _NE;
	case '>' :
		if (asExpr[aiPos + 1] == '=') 
		{
			aiPos += 2;
			return _GE;
		}	
		aiPos ++;
		return _GT;
	case '<' :
		if (asExpr[aiPos + 1] == '=') 
		{
			aiPos += 2;
			return _LE;
		}	
		aiPos ++;
		return _LT;
	default :
		if (asExpr[aiPos]=='\"' || asExpr[aiPos]=='\'') 
		{
			aiPos++;
			string literal;
			for( ; aiPos < asExpr.size(); aiPos++ )
			{
				char ch = asExpr[aiPos]; 
				switch( ch )
				{
				case '\\':
					aiPos++;
					if( aiPos < asExpr.size() )
					{
						literal += asExpr[aiPos];
					}//if
					break;
				case '\"':
				case '\'':
					goto l_token;
				default:
					literal += ch;
				}//switch
			}//for
l_token:
			litToken.name = literal;
			litToken.type = _LITERAL;
			aiPos++;
			return _LITERAL;
		}
		if (isalnum(asExpr[aiPos]) || asExpr[aiPos]=='.' || asExpr[aiPos]=='*' || asExpr[aiPos]=='/' || asExpr[aiPos]=='@' || asExpr[aiPos]=='_' || asExpr[aiPos]=='-') 
		{
			string::size_type res = aiPos;	
			string::size_type stpos = 0;
			while (1) 
			{
				SkipWhiteSpaces(asExpr, res);
				stpos = res;
				l_repeat:
				for(; res<asExpr.size() && asExpr[res]!='$' && (isalnum(asExpr[res]) || asExpr[res]=='.' || asExpr[res]=='*' || asExpr[res]=='/' || asExpr[res]=='@'
					|| asExpr[res]=='_' || asExpr[res]=='-'); res++);
				//if(res >= asExpr.size())
				//	break;
				switch( res < asExpr.size() ? asExpr[res] : '\0' )
				{
				case '!':
				case '(':
					{
						string token = asExpr.substr(stpos, res - stpos);
						if(asExpr[res] == '(')
						{
							if(CompareStr(token.c_str(), "and")) break;
							if(CompareStr(token.c_str(), "or")) break;
							if(CompareStr(token.c_str(), "not")) break;
						}
						
						if(CompareStr(token.c_str(), "date")) 
						{
							string::size_type i = asExpr.find(")", res);
							
							if (i != string::npos) 
							{
								string sdate = asExpr.substr(res + 1, i - res - 1);
								litToken.name = sdate;
								litToken.type = _DATE;
								aiPos = i + 1;
								return _DATE;
							}
							else return _NODE;
						}
						
						if(CompareStr(token.c_str(), "index")) litToken.listMethod.push_back(M_INDEX);
						if(CompareStr(token.c_str(), "value")) litToken.listMethod.push_back(M_VALUE);
						if(CompareStr(token.c_str(), "nodeType")) litToken.listMethod.push_back(M_NODETYPE);
						if(CompareStr(token.c_str(), "nodeName")) litToken.listMethod.push_back(M_NODENAME);
						if(CompareStr(token.c_str(), "end")) litToken.listMethod.push_back(M_END);
						if(CompareStr(token.c_str(), "number")) litToken.listMethod.push_back(M_NUMBER);

						if (litToken.name.empty() && !litToken.listMethod.size()) 
						{
							string::size_type iBr = asExpr.find(")", stpos);
							if (iBr != string::npos) 
							{
								litToken.name = asExpr.substr(stpos, iBr - stpos + 1);
								res = iBr;
							}
							else
							{
								litToken.name = token;
							}
						}

						res++;
						continue;
						break;
					}
				case ')':
					{
						string token = asExpr.substr(stpos, res - stpos);
						aiPos = res;
						//define what's this
						if (litToken.name.empty() && !litToken.listMethod.size()) 
							litToken.name = token;
						//			sTokenName = s;
						if(isdigit(litToken.name[0]))
						{
							return _NUMERIC;
						}
						else
						{
							return _NODE;
						}//else
					}
//					res++;
//					continue;
//					break;
				case '[':
					{					
						NextSquareBr(asExpr, res);
						res++;
						goto l_repeat;
					}
				default:
					string token = asExpr.substr(stpos, res - stpos);
					aiPos = res;
					//define what's this
					if(CompareStr(token.c_str(), "and")) return _AND;
					if(CompareStr(token.c_str(), "or")) return _OR;
					if(CompareStr(token.c_str(), "not")) return _NOT;
					
					if (litToken.name.empty() && !litToken.listMethod.size()) 
						litToken.name = token;
					//			sTokenName = s;
					if(isdigit(litToken.name[0]))
						return _NUMERIC;
					break;
				}
				
				break;
			}
			if(litToken.name.empty())
				litToken.name = ".";
			return _NODE;
		}
	}
	aiPos++;
	return _NONE;
}

void CFilterExpr::Parse(const string& asExpr)
{
	string::size_type pos = 0;
	EXPRTOKEN token = _OPENBR;
	Literal litToken;
	Push(token, litToken);
	while( (token = NextToken(asExpr, pos, litToken)) != _NONE)
	{
		Push(token, litToken); 	
	}
	while(GetPop() != _OPENBR)
		Pop();
	if(m_Parsed.size() == 1 && m_LitTable.size() == 1)
	{
		if(m_LitTable[0].type == _NUMERIC)
		{
			char buffer[32] = { 0 };
			//_itoa(m_SuccIndex, buffer, 10);
			sprintf(buffer, "%d", m_SuccIndex);
			m_LitTable.push_back(Literal(string(buffer), _NUMERIC));
			m_Parsed.push_back(101);
			m_Parsed.push_back(_EQ);
		}
		else
		{
			m_Parsed.push_back(1);
			m_Parsed.push_back(_AND);
		}
	}

}

LIST_IT CFilterExpr::GetNextOperator()
{
	m_SetOperator = _ANY;
	LIST_IT it=m_Parsed.begin();
	for(; it != m_Parsed.end();)
	{
		if(*it>=10 && *it<100) 
		{
			if(*it != _ANY && *it != _ALL)
				return it;
			if(*it ==_ALL)
				m_SetOperator = _ALL;
			it = m_Parsed.erase(it);  
		}
		else
		{	
			it++;
		}
	}
	return m_Parsed.end();
}

bool CFilterExpr::BoolEvaluate(int operand)
{
	if(operand < 10 ) return operand == 1;
	if(operand>=100)
	{
		XMLDOM_Node *node;
		if (m_Node->GetNodeType() == NODE_ELEMENT && static_cast<XMLDOM_Element *>(m_Node)->SelectSingleNode(m_LitTable[operand - 100].name, node))
		{
			node->Release();
			return true;
		}
	}
	return false;
}

bool CFilterExpr::StringEvaluate(LIST_IT itOper, int l_operand, int r_operand)
{
	bool bResult;

	_VARIANT rVar, lVar;
	if (m_LitTable[r_operand - 100].type != _NODE) 
	{
		switch(m_LitTable[r_operand - 100].type) 
		{
		case _NUMERIC:
			rVar.iVal = atoi(m_LitTable[r_operand - 100].name.c_str());
			rVar.varType = _VARIANT::_INT;
			break;
		case _DATE:
			rVar = _VARIANT(&(m_LitTable[r_operand - 100].name), _VARIANT::_DATE, m_Node);
			break;
		case _LITERAL:
			rVar.sVal = m_LitTable[r_operand - 100].name.c_str();
			rVar.varType = _VARIANT::_STR;
			break;
		default:
			break;
		}
	}
	else
	{
		XMLDOM_Node *node;
		if(m_Node->GetNodeType() == NODE_ELEMENT && static_cast<XMLDOM_Element *>(m_Node)->SelectSingleNode(m_LitTable[r_operand - 100].name, node))
		{
			ExecuteMethods(node, m_LitTable[r_operand - 100], rVar);

			node->Release();
		}
		else
		{
			rVar.Clear();
		}
	}

	if (m_LitTable[l_operand - 100].type != _NODE) 
	{
		switch(m_LitTable[l_operand - 100].type) 
		{
		case _NUMERIC:
			lVar.iVal = atoi(m_LitTable[l_operand - 100].name.c_str());
			lVar.varType = _VARIANT::_INT;
			break;
		case _DATE:
			lVar = _VARIANT(&(m_LitTable[l_operand - 100].name), _VARIANT::_DATE, m_Node);
			break;
		case _LITERAL:
			lVar.sVal = m_LitTable[l_operand - 100].name.c_str();
			lVar.varType = _VARIANT::_STR;
			break;
		default:
			break;
		}
		bResult = EvalStrOperator(itOper, lVar, rVar);
	}
	else
	{
		bResult = EvalNodeOperators(itOper, l_operand, lVar, rVar);
	}

	return bResult;
}


bool CFilterExpr::EvalStrOperator(LIST_IT itOper, _VARIANT &lVar, _VARIANT &rVar)
{
	if(rVar.varType == _VARIANT::_UNDEF || lVar.varType == _VARIANT::_UNDEF) return false;
	switch(rVar.varType) 
	{
	case _VARIANT::_STR:
		lVar.ToStr();
		switch(*itOper)
		{
		case _EQ:
			return !(lVar.sVal.compare(rVar.sVal));
			break;
		case _IEQ:
			return CompareStr(lVar.sVal.c_str(), rVar.sVal.c_str());
			break;
		case _NE:
			return lVar.sVal.compare(rVar.sVal) != 0;
			break;
		case _INE:
			return !CompareStr(lVar.sVal.c_str(), rVar.sVal.c_str());
			break;
		case _LT:
			return lVar.sVal.compare(rVar.sVal) <0;
			break;
		case _ILT:
			return CompareString(lVar.sVal.c_str(), rVar.sVal.c_str()) <0;
			break;
		case _GT:
			return lVar.sVal.compare(rVar.sVal) >0;
			break;
		case _IGT:
			return CompareString(lVar.sVal.c_str(), rVar.sVal.c_str()) >0;
			break;
		case _LE:
			return lVar.sVal.compare(rVar.sVal) <=0;
			break;
		case _ILE:
			return CompareString(lVar.sVal.c_str(), rVar.sVal.c_str()) <=0;
			break;
		case _GE:
			return (lVar.sVal.compare(rVar.sVal)) >=0;
			break;
		case _IGE:
			return CompareString(lVar.sVal.c_str(), rVar.sVal.c_str()) >=0;
			break;
		}
		return false;
		break;
	case _VARIANT::_INT:
		lVar.ToInt();
		switch(*itOper)
		{
		case _EQ:
		case _IEQ:
			return lVar.iVal == rVar.iVal;
			break;
		case _NE:
		case _INE:
			return lVar.iVal != rVar.iVal;
			break;
		case _LT:
		case _ILT:
			return lVar.iVal < rVar.iVal;
			break;
		case _GT:
		case _IGT:
			return lVar.iVal > rVar.iVal;
			break;
		case _LE:
		case _ILE:
			return lVar.iVal <= rVar.iVal;
			break;
		case _GE:
		case _IGE:
			return lVar.iVal >= rVar.iVal;
			break;
		}
		return false;
		break;
	case _VARIANT::_DATE: {
		lVar.ToDate();
		if(lVar.varType != _VARIANT::_DATE) return false;
		int timecmp = lVar.dateVal.tm_year - rVar.dateVal.tm_year;
		if(!timecmp)
			timecmp = lVar.dateVal.tm_mon - rVar.dateVal.tm_mon;
		if(!timecmp)
			timecmp = lVar.dateVal.tm_mday - rVar.dateVal.tm_mday;
		switch(*itOper)
		{
		case _EQ:
		case _IEQ:
			return timecmp == 0;
			break;
		case _NE:
		case _INE:
			return timecmp != 0;
			break;
		case _LT:
		case _ILT:
			return timecmp < 0;
			break;
		case _GT:
		case _IGT:
			return timecmp > 0;
			break;
		case _LE:
		case _ILE:
			return timecmp <= 0;
			break;
		case _GE:
		case _IGE:
			return timecmp >= 0;
			break;
		}
		return false;
		break;};
	default:
		return false;
	}
	return false;
}

bool CFilterExpr::EvalNodeOperators(LIST_IT itOper, int l_operand, _VARIANT &lVar, _VARIANT &rVar)
{
	bool bResult = false;

	if (m_Node->GetNodeType() == NODE_ELEMENT)
	{
		XMLDOM_NodeList *list = NULL;
		static_cast<XMLDOM_Element *>(m_Node)->SelectNodes(m_LitTable[l_operand - 100].name, list);
		
		do
		{
			XMLDOM_NodeList::size_type nItemCount = list->GetLength();
			if (!nItemCount)
			{
				break;
			}
			
			XMLDOM_NodeList::size_type nItemIndex = 0;
			for (; nItemIndex != nItemCount; ++nItemIndex)
			{
				XMLDOM_Node *node;
				list->GetItem(nItemIndex, node);
				
				ExecuteMethods(node, m_LitTable[l_operand - 100], lVar);
				
				node->Release();
				
				bool bSetOperatorAll = m_SetOperator == _ALL;
				if (bSetOperatorAll != EvalStrOperator(itOper, lVar, rVar))
				{
					bResult = !bSetOperatorAll;
					break;
				}
			}
			if (nItemIndex != nItemCount)
			{
				break;
			}

			bResult = m_SetOperator == _ALL;
		}
		while (false);

		list->Release();
	}
	
	return bResult;
}

bool CFilterExpr::EvaluateExpr(LIST_IT itOper, LIST_IT itBegin)
{
	LIST_IT tempIt = itOper;
	int rightOp = _NONE, leftOp = _NONE;
	
	if(tempIt != itBegin)
	{
		rightOp = *(--tempIt);
	}
	if(tempIt != itBegin)
	{
		leftOp = *(--tempIt);
	}

	switch(*itOper) 
	{
	case _AND:
		return (BoolEvaluate(leftOp) && BoolEvaluate(rightOp));
		break;
	case _OR:
		return (BoolEvaluate(leftOp) || BoolEvaluate(rightOp));
		break;
	case _NOT:
		return !BoolEvaluate(rightOp);
		break;
	}
	return StringEvaluate(itOper, leftOp, rightOp);
}

void CFilterExpr::InsertResult(LIST_IT itOper, int iRes)
{
	LIST_IT itStart = itOper;
	itStart--;
	if (*itOper != _NOT) 
	{
		itStart--;
	}
	*itOper = iRes;
	m_Parsed.erase(itStart, itOper);
}

bool CFilterExpr::Evaluate()
{
	bool bRetv = true;
	LIST_IT itOper;

	if(m_Parsed.size())
	{
		while ((itOper = GetNextOperator()) != m_Parsed.end()) 
		{
			InsertResult(itOper, EvaluateExpr(itOper, m_Parsed.begin()));
		}

		bRetv = *m_Parsed.begin() == 1;  
	}

	return bRetv;
}

void CFilterExpr::ExecuteMethods(XMLDOM_Node *pNode, Literal& litToken, _VARIANT& Var)
{
	if (!litToken.listMethod.size()) 
	{
		string sValue;
		pNode->GetText(sValue);
		Var = _VARIANT(&sValue, _VARIANT::_STR);
		return;
	}

	for(unsigned int i=0; i<litToken.listMethod.size();i++) 
	{
		switch(litToken.listMethod[i])
		{
			case M_NODENAME:
			{
				string sValue;
				pNode->GetTitle(sValue);
				Var = _VARIANT(&sValue, _VARIANT::_STR);
				break;
			}
			case M_NODETYPE:
			{
				int tmp = pNode->GetNodeType();
				Var = _VARIANT(&tmp, _VARIANT::_INT);
				break;
			}
			case M_NUMBER:
			{
				int tmp = atoi(Var.sVal.c_str());
				Var = _VARIANT(&tmp, _VARIANT::_INT);
				break;
			}
			case M_INDEX:
			{
				Var = _VARIANT(&m_SuccIndex, _VARIANT::_INT);
				break;
			}
			case M_VALUE:
			case M_END: 
			default:
			{
				string sValue;
				pNode->GetText(sValue);
				Var = _VARIANT(&sValue, _VARIANT::_STR);
				break;
			}
		}
	}
}

bool NextSquareBr(const string& asExpr, string::size_type &aiPos)
{	
	int nNumSqBr = 0;
	for( ; aiPos < asExpr.size(); aiPos++ )
	{
		char ch = asExpr[aiPos]; 
		switch( ch )
		{
		case '\\':
			aiPos++;			
			break;
		case '\"':
		case '\'':			
			NextQuote(asExpr, aiPos);
			break;
		case '[':
			nNumSqBr++;
			break;
		case ']':
			//aiPos++;
			nNumSqBr--;
			if(!nNumSqBr)
				goto lRet;
		}//switch
	}//for
lRet:
	return true;
}

bool NextQuote(const string& asExpr, string::size_type &aiPos)
{	
	aiPos++;
	for( ; aiPos < asExpr.size(); aiPos++ )
	{
		char ch = asExpr[aiPos]; 
		switch( ch )
		{
		case '\\':
			aiPos++;			
			break;
		case '\"':
		case '\'':
			goto lReturn;
		}//switch
	}//for
lReturn:
	return true;
}
