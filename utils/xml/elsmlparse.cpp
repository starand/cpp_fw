#include "StdAfx.h"
#include "XMLParserLite.h"
#include "strutils.h"
#include "elxmlparse.h"
#include "strutils.h"

using namespace StrUtils;

#define BIT_NAME		((char)'\x01')
#define BIT_NAME_FIRST	((char)'\x02')

void CXMLParserStringCache::TranslateString(const char *szBegin, const char *szEnd, string &sResult)
{
	assert(szBegin < szEnd);
	
	unsigned char chBegin = *szBegin;
	unsigned char chEnd = *(szEnd - 1);
	
	unsigned int uCacheIndex = 0xFF & (chBegin ^ (chEnd << 4) ^ (chEnd >> 4));
	
	string &sCacheElement = m_asCache[uCacheIndex];
	
	if (sCacheElement.compare(0, string::npos, szBegin, szEnd - szBegin) != 0)
	{
		sCacheElement.assign(szBegin, szEnd - szBegin);
	}
	
	sResult = sCacheElement;
}

////////////////////////////////////////////////////////////////////////////////

CXMLParser::CXMLParser():
	m_uErrorPosition(0),
	m_iParseResult(RESULT_OK)
{
}


CXMLParser::~CXMLParser(){
}


int CXMLParser::ParseDocument(const string &sXMLText, void *fpFilter, void *pvContext, XMLDOM_Document *pdDocument)
{
	int iResult = RESULT_OK;
	const char *szBegin = sXMLText.c_str(), *szEnd = szBegin + sXMLText.size(), *szResult;
	do 
	{
		// check BOM on start string
		if(*((unsigned short*)szBegin) == 0xFEFF || *((unsigned short*)szBegin) == 0xFFFE)
		{
			// utf-16
			iResult = RESULT_NOT_SUPPORTED_CHARSET;
			break;
		}
		if(sXMLText.size() >= 3 && ((unsigned char*)szBegin)[0] == 0xEF && ((unsigned char*)szBegin)[1] == 0xBB &&
			((unsigned char*)szBegin)[2] == 0xBF)
		{
			szBegin += 3;
		}

		CParseContext pcContext(pdDocument, fpFilter, pvContext, DOM_LEVEL_ROOT);
		iResult = InternalParseDocument(szBegin, szEnd, szResult, pcContext);
		m_uErrorPosition = (unsigned int)(szResult - szBegin);
	}
	while (false);

	m_iParseResult = iResult;
	return iResult;
}

int CXMLParser::StringToNode(const string &sXMLText, XMLDOM_Document *pdDocument, XMLDOM_Node *&pnOutNode)
{
	const char *szBegin = sXMLText.c_str(), *szEnd = szBegin + sXMLText.size(), *szResult;
	CParseContext pcContext(pdDocument, NULL, NULL, DOM_LEVEL_ROOT);
	int iResult = RecognizeAndBuildNode(szBegin, szEnd, false, szResult, pcContext, pnOutNode);

	m_iParseResult = iResult;
	m_uErrorPosition = (unsigned int)(szResult - szBegin);
	return iResult;
}

// Begin: can be space
// Result: meaningless
int CXMLParser::InternalParseDocument(const char *szBegin, const char *szEnd, const char *&szResult, CParseContext &pcContext) 
{
/*
 *	Though the function takes pointers to begin and end there still must be 
 *	a '\0' terminator beyond szEnd.
 */
	if (!pcContext.m_pdDocument) 
	{
		assert(false && "No XML pcContext.m_pdDocument passed to InternalParseDocument");
		return RESULT_INTERNAL_ERROR;
	}

	pcContext.m_pdDocument->Clear();

	const char *szNewBegin;
	StripSpaces(szBegin, szEnd, szNewBegin);
	szBegin = szNewBegin;

	int iResult = RESULT_OK;

	if (IsStartOfDocDefinition(szBegin, szEnd, szNewBegin))
	{
		szBegin = szNewBegin;
		
		iResult = ParseDocDefinition(szBegin, szEnd, szResult, pcContext);
		if (iResult == RESULT_OK)
		{
			szBegin = szResult;
		}
	}

	if (iResult == RESULT_OK)
	{
		bool bWasAnyElement = false;

		while (szBegin < szEnd)
		{
			const char *szInitialBegin = szBegin;

			XMLDOM_Node *pnNode;
			iResult = RecognizeAndBuildNode(szBegin, szEnd, false, szResult, pcContext, pnNode);
			if (iResult != RESULT_SKIPPED)
			{
				if (iResult != RESULT_OK)
				{
					break;
				}

				if (!pnNode)
				{
					// Everything OK, just some spaces after end of XML
					break;
				}

				DOMNodeType ntNodeType = pnNode->GetNodeType();
				assert(ntNodeType != NODE_INVALID);

				if (ntNodeType == NODE_ELEMENT)
				{
					if (!bWasAnyElement)
					{
						bWasAnyElement = true;
					}
					else
					{
						pnNode->Release();

						szResult = szInitialBegin;
						iResult = RESULT_ONLY_ONE_ROOT_ELEMENT_ALLOWED;
						break;
					}
				}

				pcContext.m_pdDocument->AppendChildNode(pnNode);
				pnNode->Release();
			}

			szBegin = szResult;
		}

		if (iResult == RESULT_OK)
		{
			if (!bWasAnyElement) 
			{
				szResult = szBegin;
				iResult = RESULT_TOP_LEVEL_ELEMENT_REQUIRED;
			}
		}
	}

	return iResult;
}

// Begin: must be space
// Result: not trimmed
int CXMLParser::ParseDocDefinition(const char *szBegin, const char *szEnd, const char *&szResult, CParseContext &pcContext)
{
	const char *szInitialBegin = szBegin;

	bool bVersionAttrPresent = false;
	bool bEncodingAttrPresent = false;
	string sAttrName, sAttrValue;

	int iResult;

	// This block of code could be implemented with ParseTagHeaderAttributes()
	// but let it be to avoid using intermediate vectors
	while ((iResult = ParseAttribute(szBegin, szEnd, false, szResult, sAttrName, sAttrValue)) == RESULT_OK)
	{
		if (!bVersionAttrPresent)
		{
			if (CompareStr(sAttrName.c_str(), szXMLPparser_AttrVersion))
			{
				bVersionAttrPresent = true; 
			}
			else
			{
				szResult = szBegin;
				iResult = RESULT_SPECIFIED_ATTRIBUTE_UNEXPECTED_AT_POS;
				break;
			}
		}

		if (!bEncodingAttrPresent)
		{
			if (CompareStr(sAttrName.c_str(), "encoding"))
			{
				if (CompareStr(sAttrValue.c_str(), szXMLPparser_UTF8_Encoding))
				{
					pcContext.m_pdDocument->SetCharset(ECS_UTF8);
				}
				else if (CompareStr(sAttrValue.c_str(), szXMLPparser_UTF16_Encoding))
				{
					szResult = szBegin;
					iResult = RESULT_NOT_SUPPORTED_CHARSET;
					break;
				}
				bEncodingAttrPresent = true;
			}
		}

		if (!pcContext.m_fpFilter || ((XMLDOM_FilterProc)pcContext.m_fpFilter)(sAttrName, NODE_ATTRIBUTE, DOM_LEVEL_ROOT, pcContext.m_pvContext))
		{
			pcContext.m_pdDocument->SetAttribute(sAttrName, sAttrValue);
		}

		szBegin = szResult;
		if (szBegin >= szEnd) 
		{
			iResult = RESULT_UNEXPECTED_END_OF_FILE;
			break;
		}
	}

	if (iResult == RESULT_OUT_OF_DATA) 
	{
		szBegin = szResult;

		if (bVersionAttrPresent)
		{
			if (IsEndOfDocDefinition(szBegin, szEnd, szResult))
			{
				iResult = RESULT_OK;
			}
			else
			{
				szResult = szBegin;
				return RESULT_INVALID_SYNTAX_FOR_XML_DECL;
			}
		}
		else
		{
			szResult = szInitialBegin;
			iResult = RESULT_INVALID_SYNTAX_FOR_XML_DECL;
		}
	}

	return iResult;
}

// Begin: can be space
// Result: trimmed if close tag, not trimmed in other cases
int CXMLParser::RecognizeAndBuildNode(const char *szBegin, const char *szEnd, bool bSkipped, const char *&szResult, CParseContext &pcContext, XMLDOM_Node *&pnOutNode)
{
	pnOutNode = NULL;
	int iResult;

	const char *szNewBegin;
	StripSpaces(szBegin, szEnd, szNewBegin);
	szBegin = szNewBegin;
	
	if (szBegin < szEnd)
	{
		if (IsStartOfComment(szBegin, szEnd, szNewBegin)) 
		{
			szBegin = szNewBegin;
			
			string sData;
			if ((iResult = ParseComment(szBegin, szEnd, bSkipped, szResult, sData)) == RESULT_OK) 
			{
				if (!bSkipped && (!pcContext.m_fpFilter || ((XMLDOM_FilterProc)pcContext.m_fpFilter)(sData, NODE_COMMENT, pcContext.m_nLevel, pcContext.m_pvContext))) 
				{
					pcContext.m_pdDocument->CreateComment(sData, (XMLDOM_Comment *&)pnOutNode);
					iResult = RESULT_OK;
				}
				else
				{
					iResult = RESULT_SKIPPED;
				}
			}
		} 
		else if (IsStartOfCDATASection(szBegin, szEnd, szNewBegin)) 
		{
			szBegin = szNewBegin;
			
			string sData;
			if ((iResult = ParseCDATASection(szBegin, szEnd, bSkipped, szResult, sData)) == RESULT_OK)
			{
				if (!bSkipped && (!pcContext.m_fpFilter || ((XMLDOM_FilterProc)pcContext.m_fpFilter)(sData, NODE_CDATA_SECTION, pcContext.m_nLevel, pcContext.m_pvContext)))
				{
					pcContext.m_pdDocument->CreateCDataSection(sData, (XMLDOM_CDATASection *&)pnOutNode);
					iResult = RESULT_OK;
				}
				else
				{
					iResult = RESULT_SKIPPED;
				}
			}
		}
		else if (IsTextFragment(szBegin, szEnd)) 
		{
			string sData;
			if ((iResult = ParseTextFragment(szBegin, szEnd, bSkipped, szResult, sData)) == RESULT_OK)
			{
				if (!bSkipped && (!pcContext.m_fpFilter || ((XMLDOM_FilterProc)pcContext.m_fpFilter)(sData, NODE_TEXT, pcContext.m_nLevel, pcContext.m_pvContext)))
				{
					pcContext.m_pdDocument->CreateTextNode(sData, (XMLDOM_Text *&)pnOutNode);
					iResult = RESULT_OK;
				}
				else
				{
					iResult = RESULT_SKIPPED;
				}
			}
		} 
		else if (IsCloseTag(szBegin, szEnd, szNewBegin))
		{
			szResult = szBegin;
			iResult = RESULT_CLOSE_TAG;
		}
		else
		{
			iResult = BuildElement(szBegin, szEnd, bSkipped, szResult, pcContext, (XMLDOM_Element *&)pnOutNode);
		}
	}
	else
	{
		szResult = szBegin;
		iResult = !bSkipped ? RESULT_OK : RESULT_SKIPPED;
	}

	return iResult;
}

// Begin: must not be space
// Result: not trimmed
int CXMLParser::BuildElement(const char *szBegin, const char *szEnd, bool bSkipped, const char *&szResult, 
							 CParseContext &pcContext, XMLDOM_Element *&peOutElement)
{
	const char *szInitialBegin = szBegin;

	int iResult;

	bool bEmptyTag;
	vector<string> vsAttrNames, vsAttrValues; 
	string sTagName, sTagNamespace;
	// ParseTagHeader does not clear vectors so they must be empty on entry
	if ((iResult = ParseTagHeader(szBegin, szEnd, bSkipped, szResult, sTagName, sTagNamespace,
		vsAttrNames, vsAttrValues, bEmptyTag)) == RESULT_OK)
	{
		szBegin = szResult;

		if (!bSkipped && (!pcContext.m_fpFilter || ((XMLDOM_FilterProc)pcContext.m_fpFilter)(sTagName, NODE_ELEMENT, pcContext.m_nLevel, pcContext.m_pvContext))) 
		{
			pcContext.m_pdDocument->CreateElement(sTagName, sTagNamespace, peOutElement);
		} 
		else 
		{
			bSkipped = true;
		}

		pcContext.m_nLevel += 1;

		XMLDOM_Element *peElement = peOutElement;
		
		if (!bSkipped) 
		{
			assert(vsAttrNames.size() == vsAttrValues.size());

			for (unsigned int nAttrIndex = 0; nAttrIndex < vsAttrNames.size(); ++nAttrIndex)
			{
				const string &sAttribute = vsAttrNames[nAttrIndex];

				if (!pcContext.m_fpFilter || ((XMLDOM_FilterProc)pcContext.m_fpFilter)(sAttribute, NODE_ATTRIBUTE, pcContext.m_nLevel, pcContext.m_pvContext)) 
				{
					const string &sValue = vsAttrValues[nAttrIndex];

					peElement->SetAttribute(sAttribute, sValue);
				}
			}
		}

		if (!bEmptyTag)
		{
			do
			{
				XMLDOM_Node *pnChildNode;
				if ((iResult = RecognizeAndBuildNode(szBegin, szEnd, bSkipped, szResult, pcContext, pnChildNode)) == RESULT_OK) 
				{
					if (pnChildNode)
					{
						peElement->AppendChildNode(pnChildNode);
						pnChildNode->Release();
					}
				}
				else
				{
					assert(pnChildNode == NULL);
				}
				
				szBegin = szResult;
				if (szBegin >= szEnd)
				{
					szResult = szInitialBegin;
					iResult = RESULT_START_TAG_WO_CLOSE_TAG;
					break;
				}
			}
			while (iResult == RESULT_OK || iResult == RESULT_SKIPPED);

			if (iResult == RESULT_CLOSE_TAG)
			{
				string sCloseTagName, sCloseTagNamespace;
				if ((iResult = ParseCloseTagHeader(szBegin, szEnd, bSkipped, szResult, 
					sCloseTagName, sCloseTagNamespace)) == RESULT_OK) 
				{
					if(sTagName != sCloseTagName || sTagNamespace != sCloseTagNamespace)
					{
						szResult = szBegin;
						iResult = RESULT_CLOSE_TAG_WO_START_TAG;
					}
				}
			}

			if (iResult != RESULT_OK)
			{
				if (peElement)
				{
					peElement->Release();
				}

				peOutElement = NULL;
			}
		}

		pcContext.m_nLevel -= 1;

		if (iResult == RESULT_OK && bSkipped)
		{
			iResult = RESULT_SKIPPED;
		}
	}

	return iResult;
}

// Begin: must not be space
// Result: not trimmed
int CXMLParser::ParseTagHeader(const char *szBegin, const char *szEnd, bool bSkipped, const char *&szResult, 
	string &sTagName, string &sTagNamespace, vector<string> &vsAttrNames, vector<string> &vsAttrValues, bool &bIsEmpty)
{
	const char *szNewBegin;
	assert(!IsCloseTag(szBegin, szEnd, szNewBegin));
	bool bIsLT = IsLT(szBegin, szEnd, szNewBegin);
	assert(bIsLT);
	szBegin = szNewBegin;
	
	int iResult;
	
	if ((iResult = ParseTagHeaderName(szBegin, szEnd, bSkipped, szResult, sTagName, sTagNamespace)) == RESULT_OK)
	{
		szBegin = szResult;
		
		if (!IsSpace(*szBegin) || (iResult = ParseTagHeaderAttributes(szBegin, szEnd, bSkipped, szResult, vsAttrNames, vsAttrValues)) == RESULT_OUT_OF_DATA)
		{
			szBegin = szResult;

			bIsEmpty = false;
			iResult = RESULT_OK; // Let it be for safety -- actually it is really necessary here (see current "if")
			if (IsEmptyTagGT(szBegin, szEnd, szResult))
			{
				bIsEmpty = true;
			}
			else if (!IsGT(szBegin, szEnd, szResult))
			{
				szResult = szBegin;
				iResult = RESULT_ERROR_PARSING_TAG_HEADER;
			}
		}
	}

	return iResult;
}	

// Begin: must not be space
// Result: not trimmed
int CXMLParser::ParseCloseTagHeader(const char *szBegin, const char *szEnd, bool bSkipped, const char *&szResult, 
	string &sTagName, string &sTagNamespace)
{
	const char *szNewBegin;
	bool bIsLT = IsCloseTag(szBegin, szEnd, szNewBegin);
	assert(bIsLT);
	szBegin = szNewBegin;

	int iResult;

	if ((iResult = ParseTagHeaderName(szBegin, szEnd, bSkipped, szResult, sTagName, sTagNamespace)) == RESULT_OK)
	{
		szBegin = szResult;

		StripSpaces(szBegin, szEnd, szNewBegin);
		szBegin = szNewBegin;
		
		if (szBegin < szEnd)
		{
			iResult = RESULT_OK; // let it be for safety
			if (!IsGT(szBegin, szEnd, szResult))
			{
				szResult = szBegin;
				iResult = RESULT_ERROR_PARSING_TAG_HEADER;
			}
		}
		else
		{
			szResult = szBegin;
			iResult = RESULT_UNEXPECTED_END_OF_FILE;
		}
	}
	
	return iResult;
}	

// Begin: must not be space
// Result: not trimmed
int CXMLParser::ParseTagHeaderName(const char *szBegin, const char *szEnd, bool bSkipped, const char *&szResult, 
	string &sTagName, string &sTagNamespace)
{
	int iResult;
	
	if (szBegin < szEnd)
	{
		char ch = *szBegin;
		if (!IsSpace(ch)) 
		{
			const char *szTagBegin = szBegin;
			bool bFirstChar = true;
			if (IsValidXMLNameFirtsChar(ch))
			{
				++szBegin;
				if (szBegin < szEnd) 
				{
					bFirstChar = false;
					ch = *szBegin;

					while (IsValidXMLNameChar(ch))
					{
						++szBegin;
						if (szBegin >= szEnd) 
						{
							break;
						}
						
						ch = *szBegin;
					}
				}
			}
			
			if (szBegin < szEnd)
			{
				if (!bFirstChar)
				{
					m_scStringCache.TranslateString(szTagBegin, szBegin, sTagName);
					
					string::size_type nPos = sTagName.find(":");
					if(nPos != string::npos)
					{
						sTagNamespace = sTagName.substr(0, nPos);
						sTagName.erase(0, nPos + 1);
					}

					szResult = szBegin;
					iResult = RESULT_OK;
				}
				else
				{
					szResult = szBegin;
					iResult = RESULT_TAG_NAME_INVALID;
				}
			}
			else
			{
				szResult = szBegin;
				iResult = RESULT_UNEXPECTED_END_OF_FILE;
			}
		}
		else
		{
			szResult = szBegin;
			iResult = RESULT_WHITESPACE_NOT_ALLOWED_AT_POS;
		}
	}
	else
	{
		szResult = szBegin;
		iResult = RESULT_UNEXPECTED_END_OF_FILE;
	}
	
	return iResult;
}

// Begin: must be space
// Result: trimmed
int CXMLParser::ParseTagHeaderAttributes(const char *szBegin, const char *szEnd, bool bSkipped, const char *&szResult, 
	vector<string> &vsAttrNames, vector<string> &vsAttrValues)
{
	assert(IsSpace(*szBegin));

	string sAttrName, sAttrValue;

	int iResult;
	
	while ((iResult = ParseAttribute(szBegin, szEnd, bSkipped, szResult, sAttrName, sAttrValue)) == RESULT_OK) 
	{
		if (!bSkipped) 
		{
			vsAttrNames.push_back(sAttrName);
			vsAttrValues.push_back(sAttrValue);
		}
		
		szBegin = szResult;
		if (szBegin >= szEnd) 
		{
			iResult = RESULT_UNEXPECTED_END_OF_FILE;
			break;
		}
	}

	return iResult;
}

// Begin: must be space for attribute or other character for out of data
// Result: not trimmed
int CXMLParser::ParseAttribute(const char *szBegin, const char *szEnd, bool bSkipped, const char *&szResult, 
	string &sAttrName, string &sAttrValue)
{
	int iResult;

	const char *szNewBegin;
	StripSpaces(szBegin, szEnd, szNewBegin);
	if (szBegin != szNewBegin)
	{
		szBegin = szNewBegin;

		if (szBegin < szEnd)
		{
			char ch = *szBegin;
			if (IsValidXMLNameFirtsChar(ch))
			{
				const char *szNameBegin = szBegin;
				
				do
				{
					++szBegin;

					if (szBegin >= szEnd)
					{
						break;
					}

					ch = *szBegin;
				}
				while (IsValidXMLNameChar(ch));

				if (szBegin < szEnd)
				{
					if (!bSkipped)
					{
						m_scStringCache.TranslateString(szNameBegin, szBegin, sAttrName);
					}

					StripSpaces(szBegin, szEnd, szNewBegin);
					szBegin = szNewBegin;

					if (szBegin < szEnd)
					{
						if (IsAttrValueDelimiter(*szBegin))
						{
							++szBegin;

							StripSpaces(szBegin, szEnd, szNewBegin);
							szBegin = szNewBegin;

							if (szBegin < szEnd)
							{
								if (IsQuote(*szBegin))
								{
									++szBegin;

									if (szBegin < szEnd)
									{
										if (!bSkipped)
										{
											sAttrValue.erase();
										}
										
										iResult = RESULT_OK;
										const char *szValueBegin = szBegin;
										while (true)
										{
											char ch = *szBegin;
											if (!IsQuote(ch))
											{
												++szBegin;

												if (IsAmpPrefix(ch)) 
												{
													char chConverted;
													if ((iResult = ConvertSymbolFromSpec(szBegin, szEnd, szResult, chConverted)) != RESULT_OK) 
													{
														break;
													}

													if (!bSkipped) 
													{
														sAttrValue.append(szValueBegin, szBegin - 1);
														sAttrValue.append(1, chConverted); // No proper reservation is done. It's rare case when lots of chars are encoded.
													}
													
													szBegin = szResult;
													szValueBegin = szBegin;
												}

												if (szBegin >= szEnd) 
												{
													szResult = szBegin;
													iResult = RESULT_UNEXPECTED_END_OF_FILE;
													break;
												}
											}
											else
											{
												if (!bSkipped) 
												{
													sAttrValue.append(szValueBegin, szBegin);
												}

												++szBegin;
												if (szBegin < szEnd)
												{
													szResult = szBegin;
												}
												else
												{
													szResult = szBegin;
													iResult = RESULT_UNEXPECTED_END_OF_FILE;
												}

												break;
											}
										}
									}
									else
									{
										szResult = szBegin;
										iResult = RESULT_UNEXPECTED_END_OF_FILE;
									}
								}
								else
								{
									szResult = szBegin;
									iResult = RESULT_ERROR_PARSING_ATTRS;
								}
							}
							else
							{
								szResult = szBegin;
								iResult = RESULT_UNEXPECTED_END_OF_FILE;
							}
						}
						else
						{
							szResult = szBegin;
							iResult = RESULT_ERROR_PARSING_ATTRS;
						} 
					}
					else
					{
						szResult = szBegin;
						iResult = RESULT_UNEXPECTED_END_OF_FILE;
					}
				}
				else
				{
					szResult = szBegin;
					iResult = RESULT_UNEXPECTED_END_OF_FILE;
				}
			}
			else
			{
				szResult = szBegin;
				iResult = RESULT_OUT_OF_DATA;
			}
		}
		else
		{
			szResult = szBegin;
			iResult = RESULT_UNEXPECTED_END_OF_FILE;
		}
	}
	else
	{
		szResult = szBegin;
		iResult = RESULT_OUT_OF_DATA;
	}

	return iResult;
}

// Begin: meaningless
// Result: not trimmed
int CXMLParser::ParseComment(const char *szBegin, const char *szEnd, bool bSkipped, const char *&szResult, string &sData)
{
	int iResult;

	const char *szCommentEnd;
	if (SeekEndOfComment(szBegin, szEnd, szCommentEnd, szResult))
	{
		if (!bSkipped)
		{
			sData.assign(szBegin, szCommentEnd);
		}

		iResult = RESULT_OK;
	}
	else
	{
		szResult = szBegin;
		iResult = RESULT_COMMENT_WAS_NOT_CLOSED;
	}

	return iResult;
}

// Begin: meaningless
// Result: not trimmed
int CXMLParser::ParseCDATASection(const char *szBegin, const char *szEnd, bool bSkipped, const char *&szResult, string &sData)
{
	int iResult;

	const char *szCDATAEnd;
	if (SeekEndOfCDATASection(szBegin, szEnd, szCDATAEnd, szResult))
	{
		if (!bSkipped)
		{
			sData.assign(szBegin, szCDATAEnd);
		}

		iResult = RESULT_OK;
	}
	else
	{
		szResult = szBegin;
		iResult = RESULT_CDATA_SECTION_WAS_NOT_CLOSED;
	}

	return iResult;
}

// Begin: must be trimmed
// Result: trimmed
int CXMLParser::ParseTextFragment(const char *szBegin, const char *szEnd, bool bSkipped, const char *&szResult, string &sData)
{
	int iResult;
	
	const char *szInitialBegin = szBegin;
	const char *szTextEnd = szBegin; // a precaution -- can be left uninitialized
	char ch = *szBegin;
	while (true)
	{
		++szBegin;

		if (!IsAmpPrefix(ch)) 
		{
			if (!IsSpace(ch))
			{
				szTextEnd = szBegin;
			}
		} 
		else 
		{
			char chConverted;
			if ((iResult = ConvertSymbolFromSpec(szBegin, szEnd, szResult, chConverted)) != RESULT_OK) 
			{
				break;
			}
			
			if (!bSkipped) 
			{
				sData.append(szInitialBegin, szBegin - 1);
				sData.append(1, chConverted); // No proper reservation is done. It's rare case when lots of chars are encoded.
			}
			
			szBegin = szResult;
			szInitialBegin = szBegin;
			szTextEnd = szBegin;
		}
		
		if (szBegin < szEnd)
		{
			ch = *szBegin;
			if (IsTextTerminator(ch))
			{
				if (!bSkipped)
				{
					sData.append(szInitialBegin, szTextEnd);
				}

				szResult = szBegin;
				iResult = RESULT_OK;
				break;
			}
		}
		else
		{
			if (!bSkipped)
			{
				sData.append(szInitialBegin, szTextEnd);
			}
			
			szResult = szBegin;
			iResult = RESULT_OK;
			break;
		}
	}

	return iResult;
}

// Begin: must be trimmed, can be equal to end
// Result: meaningless
int CXMLParser::ConvertSymbolFromSpec(const char *szBegin, const char *szEnd, const char *&szResult, char &chConverted)
{ 
	int iResult;

	if (szBegin < szEnd) 
	{
		char ch = *szBegin;
		if (!IsAmpASCIIPrefix(ch)) 
		{
			if (IsAmpLt(szBegin, szEnd, szResult))
			{
				chConverted = '<';
				iResult = RESULT_OK;
			} 
			else if (IsAmpGt(szBegin, szEnd, szResult)) 
			{
				chConverted = '>';
				iResult = RESULT_OK;
			} 
			else if (IsAmpAmp(szBegin, szEnd, szResult)) 
			{
				chConverted = '&';
				iResult = RESULT_OK;
			} 
			else if (IsAmpApos(szBegin, szEnd, szResult)) 
			{
				chConverted = '\'';
				iResult = RESULT_OK;
			} 
			else if (IsAmpQuot(szBegin, szEnd, szResult)) 
			{
				chConverted = '"';
				iResult = RESULT_OK;
			}
			else 
			{
				szResult = szBegin;
				iResult = RESULT_INVALID_STRING;
			}
		}
		else
		{
			++szBegin;

			if (szBegin < szEnd)
			{
				ch = *szBegin;
				if (IsAmpASCIIHexPrefix(ch)) 
				{
					++szBegin;
					
					if (szBegin < szEnd)
					{
						char *szNumberEnd;
						// Here we can access memory beyond szEnd
						chConverted = char(strtol(szBegin, &szNumberEnd, 16));

						if (szNumberEnd < szEnd)
						{
							if (szNumberEnd - szBegin <= 4)
							{
								ch = *szNumberEnd;
								if (IsAmpTerminator(ch))
								{
									szResult = szNumberEnd + 1;
									iResult = RESULT_OK;
								}
								else
								{
									szResult = szNumberEnd;
									iResult = RESULT_SEMICOLON_EXPECTED;
								}
							}
							else
							{
								szResult = szBegin + 4;
								iResult = RESULT_SEMICOLON_EXPECTED;
							}
						}
						else
						{
							szResult = szEnd;
							iResult = RESULT_SEMICOLON_EXPECTED;
						}
					}
					else
					{
						szResult = szBegin;
						iResult = RESULT_SEMICOLON_EXPECTED;
					}
				}
				else 
				{
					char *szNumberEnd;
					// Here we can access memory beyond szEnd
					chConverted = char(strtol(szBegin, &szNumberEnd, 10));
					
					if (szNumberEnd < szEnd)
					{
						if (szNumberEnd - szBegin <= 5)
						{
							ch = *szNumberEnd;
							if (IsAmpTerminator(ch))
							{
								szResult = szNumberEnd + 1;
								iResult = RESULT_OK;
							}
							else
							{
								szResult = szNumberEnd;
								iResult = RESULT_SEMICOLON_EXPECTED;
							}
						}
						else
						{
							szResult = szBegin + 5;
							iResult = RESULT_SEMICOLON_EXPECTED;
						}
					}
					else
					{
						szResult = szEnd;
						iResult = RESULT_SEMICOLON_EXPECTED;
					}
				}
			}
			else
			{
				szResult = szBegin;
				iResult = RESULT_SEMICOLON_EXPECTED;
			}
		}
	}
	else
	{
		szResult = szBegin;
		iResult = RESULT_SEMICOLON_EXPECTED;
	}

	return iResult;
}


void CXMLParser::StripSpaces(const char *szBegin, const char *szEnd, const char *&szNewBegin)
{
	while (szBegin < szEnd && IsSpace(*szBegin)) 
	{
		++szBegin;
	}

	szNewBegin = szBegin;
}


bool CXMLParser::IsStartOfDocDefinition(const char *szBegin, const char *szEnd, const char *&szNewBegin)
{
	if (szEnd - szBegin >= 5 && strncmp(szBegin, "<?xml", 5) == 0)
	{
		szNewBegin = szBegin + 5;
		return true;
	}
	
	return false;
}

bool CXMLParser::IsEndOfDocDefinition (const char *szBegin, const char *szEnd, const char *&szNewBegin)
{
	if (szEnd - szBegin >= 2 && strncmp(szBegin, "?>", 2) == 0)
	{
		szNewBegin = szBegin + 2;
		return true;
	}
	
	return false;
}

bool CXMLParser::IsLT(const char *szBegin, const char *szEnd, const char *&szNewBegin)
{
	if (*szBegin == '<')
	{
		szNewBegin = szBegin + 1;
		return true;
	}

	return false;
}

bool CXMLParser::IsGT(const char *szBegin, const char *szEnd, const char *&szNewBegin)
{
	if (*szBegin == '>')
	{
		szNewBegin = szBegin + 1;
		return true;
	}
	
	return false;
}

bool CXMLParser::IsCloseTag(const char *szBegin, const char *szEnd, const char *&szNewBegin)
{
	if (szEnd - szBegin >= 2 && strncmp(szBegin, "</", 2) == 0)
	{
		szNewBegin = szBegin + 2;
		return true;
	}
	
	return false;
}

bool CXMLParser::IsEmptyTagGT(const char *szBegin, const char *szEnd, const char *&szNewBegin)
{
	if (szEnd - szBegin >= 2 && strncmp(szBegin, "/>", 2) == 0)
	{
		szNewBegin = szBegin + 2;
		return true;
	}
	
	return false;
}

bool CXMLParser::IsAmpLt(const char *szBegin, const char *szEnd, const char *&szNewBegin)
{
	if (szEnd - szBegin >= 3 && strncmp(szBegin, "lt;", 3) == 0)
	{
		szNewBegin = szBegin + 3;
		return true;
	}
	
	return false;
}

bool CXMLParser::IsAmpGt(const char *szBegin, const char *szEnd, const char *&szNewBegin)
{
	if (szEnd - szBegin >= 3 && strncmp(szBegin, "gt;", 3) == 0)
	{
		szNewBegin = szBegin + 3;
		return true;
	}
	
	return false;
}

bool CXMLParser::IsAmpAmp(const char *szBegin, const char *szEnd, const char *&szNewBegin)
{
	if (szEnd - szBegin >= 4 && strncmp(szBegin, "amp;", 4) == 0)
	{
		szNewBegin = szBegin + 4;
		return true;
	}
	
	return false;
}

bool CXMLParser::IsAmpApos(const char *szBegin, const char *szEnd, const char *&szNewBegin)
{
	if (szEnd - szBegin >= 5 && strncmp(szBegin, "apos;", 5) == 0)
	{
		szNewBegin = szBegin + 5;
		return true;
	}
	
	return false;
}

bool CXMLParser::IsAmpQuot(const char *szBegin, const char *szEnd, const char *&szNewBegin)
{
	if (szEnd - szBegin >= 5 && strncmp(szBegin, "quot;", 5) == 0)
	{
		szNewBegin = szBegin + 5;
		return true;
	}
	
	return false;
}


bool CXMLParser::IsStartOfComment(const char *szBegin, const char *szEnd, const char *&szNewBegin)
{
	if (szEnd - szBegin >= 4 && strncmp(szBegin, "<!--", 4) == 0)
	{
		szNewBegin = szBegin + 4;
		return true;
	}

	return false;
}

bool CXMLParser::SeekEndOfComment(const char *szBegin, const char *szEnd, const char *&szCommentEnd, const char *&szNewBegin)
{
	// Here we can access memory beyond szEnd
	const char *szSearchResult = strstr(szBegin, "-->");
	if (szSearchResult && szEnd - szSearchResult >= 3)
	{
		szCommentEnd = szSearchResult;
		szNewBegin = szSearchResult + 3;
		return true;
	}

	return false;
}

bool CXMLParser::IsStartOfCDATASection(const char *szBegin, const char *szEnd, const char *&szNewBegin)
{
	if (szEnd - szBegin >= 9 && strncmp(szBegin, "<![CDATA[", 9) == 0)
	{
		szNewBegin = szBegin + 9;
		return true;
	}
	
	return false;
}

bool CXMLParser::SeekEndOfCDATASection(const char *szBegin, const char *szEnd, const char *&szCDATAEnd, const char *&szNewBegin)
{
	// Here we can access memory beyond szEnd
	const char *szSearchResult = strstr(szBegin, "]]>");
	if (szSearchResult && szEnd - szSearchResult >= 3)
	{
		szCDATAEnd = szSearchResult;
		szNewBegin = szSearchResult + 3;
		return true;
	}

	return false;
}

bool CXMLParser::IsTextFragment(const char *szBegin, const char *szEnd) 
{
	return !IsTextTerminator(*szBegin);
}


bool CXMLParser::IsValidXMLNameFirtsChar(char ch)
{
	return (char)((char)(m_chCharacterKinds[(unsigned char)ch]) & BIT_NAME_FIRST) != (char)'\0'; 
}

bool CXMLParser::IsValidXMLNameChar(char ch)
{
	return (char)((char)(m_chCharacterKinds[(unsigned char)ch]) & BIT_NAME) != (char)'\0'; 
}

bool CXMLParser::IsQuote(char ch)
{
	return ch == '"';
}

bool CXMLParser::IsAttrValueDelimiter(char ch)
{
	return ch == '=';
}

bool CXMLParser::IsTextTerminator(char ch)
{
	return ch == '<';
}

bool CXMLParser::IsAmpPrefix(char ch)
{
	return ch == '&';
}

bool CXMLParser::IsAmpASCIIPrefix(char ch)
{
	return ch == '#';
}

bool CXMLParser::IsAmpASCIIHexPrefix(char ch)
{
	return (ch | (char)('x' ^ 'X')) == (char)('x' | ('x' ^ 'X'));
}

bool CXMLParser::IsAmpTerminator(char ch)
{
	return ch == ';';
}

bool CXMLParser::IsSpace(unsigned char ch)
{
	return (ch <= (unsigned char)' ');
}


const char *CXMLParser::ExplainError(int iParseResult, char *chMessageBuffer, unsigned int uErrorPosition)
{
	switch (iParseResult) 
	{
		case RESULT_OK : sprintf(chMessageBuffer, "No errors encountered"); break;
		case RESULT_ONLY_ONE_ROOT_ELEMENT_ALLOWED : sprintf(chMessageBuffer, "Only one root element allowed at pos %i", uErrorPosition); break;
		case RESULT_INVALID_NODE : sprintf(chMessageBuffer, "Invalid or unrecognized node at pos %i", uErrorPosition); break;
		case RESULT_CLOSE_TAG:
		case RESULT_CLOSE_TAG_WO_START_TAG : sprintf(chMessageBuffer, "Close tag without corresponding start tag at pos %i", uErrorPosition); break;
		case RESULT_START_TAG_WO_CLOSE_TAG : sprintf(chMessageBuffer, "Start tag without corresponding close tag at pos %i", uErrorPosition); break;
		case RESULT_ERROR_PARSING_ATTRS : sprintf(chMessageBuffer, "Error parsing attributes at pos %i", uErrorPosition); break;
		case RESULT_ERROR_PARSING_TAG_HEADER : sprintf(chMessageBuffer, "Error parsing tag header at pos %i", uErrorPosition); break;
		case RESULT_TAG_NAME_INVALID : sprintf(chMessageBuffer, "Invalid tag name at pos %i", uErrorPosition); break;
		case RESULT_DOC_DEFINITON_EXPECTED : sprintf(chMessageBuffer, "Document definition expected at pos %i", uErrorPosition); break;
		case RESULT_SPECIFIED_ATTRIBUTE_UNEXPECTED_AT_POS : sprintf(chMessageBuffer, "Specified attribute was not expected at pos %i", uErrorPosition); break;
		case RESULT_INVALID_SYNTAX_FOR_XML_DECL : sprintf(chMessageBuffer, "Invalid syntax for xml declaration at pos %i", uErrorPosition); break;
		case RESULT_TOP_LEVEL_ELEMENT_REQUIRED : sprintf(chMessageBuffer, "Top level element required at pos %i", uErrorPosition); break;
		case RESULT_WHITESPACE_EXPECTED_AT_POS : sprintf(chMessageBuffer, "Whitespace expected at pos %i", uErrorPosition); break;
		case RESULT_WHITESPACE_NOT_ALLOWED_AT_POS : sprintf(chMessageBuffer, "Whitespace not allowed at pos %i", uErrorPosition); break;
		case RESULT_SEMICOLON_EXPECTED : sprintf(chMessageBuffer, "Semicolon expected at pos %i", uErrorPosition); break;
		case RESULT_UNEXPECTED_END_OF_FILE : sprintf(chMessageBuffer, "Unexpected end of file at pos %i", uErrorPosition); break;
		case RESULT_E_NO_IMPL : sprintf(chMessageBuffer, "Not implemented at pos %i", uErrorPosition); break;
		case RESULT_CDATA_SECTION_WAS_NOT_CLOSED : sprintf(chMessageBuffer, "CDATA section was not closed at pos %i", uErrorPosition); break;
		case RESULT_COMMENT_WAS_NOT_CLOSED : sprintf(chMessageBuffer, "A comment was not closed at pos %i", uErrorPosition); break;
		case RESULT_INVALID_STRING : sprintf(chMessageBuffer, "Invalid string at pos %i", uErrorPosition); break;
		case RESULT_TAG_WAS_NOT_CLOSED : sprintf(chMessageBuffer, "Tag was not closed at pos %i", uErrorPosition); break;
		case RESULT_INTERNAL_ERROR : sprintf(chMessageBuffer, "Internal error"); break;
		case RESULT_NOT_SUPPORTED_CHARSET: sprintf(chMessageBuffer, "Invalid encoding charset"); break;
		default: sprintf(chMessageBuffer, "Unknown parse error at pos %i", uErrorPosition); break;
	}

	return chMessageBuffer;
}





//encode not valid xml chars via "&#code;"
//	- all chars with code < 32
//	- first and end space chars
//for ASCII charset
//	- all chars with code >= 128
//for utf-8 charset
//	- all sequence bytes that are not valid utf-8 sequence
//	(110xxxxx 10xxxxxx,  1110xxxx 10xxxxxx 10xxxxxx,  ...)

int ConvertStringToSpec (const string &strFrom, string &strTo, EncodingCharset nCharset)
{
	strTo.erase();
	strTo.reserve(5 * strFrom.size());
	string::size_type nHighPos = string::npos; string::size_type nHighInvPos = string::npos;
	int nLowByteCount = 0;
	string sTemp;
	bool bLowByte = false;
	unsigned char specTempNum;
	const int nBufferSize = 32;
	int nPrnSize = 0;
	char buffer[nBufferSize];

	for (string::size_type i = 0; i < strFrom.size(); ++i)
	{
		unsigned char specNum = strFrom[i];
		if(specNum < (unsigned char)'\x80')
		{
			if(nCharset == ECS_UTF8) 
			{
				nLowByteCount = 0;
				if(nHighPos != string::npos) //override bytes from last high byte and reset position
				{
					sTemp = strTo.substr(nHighPos);
					strTo.erase(nHighPos);
					for(string::size_type j = 0; j < sTemp.size(); j++)
					{
						specTempNum = sTemp[j];
						nPrnSize = snprintf(buffer, nBufferSize, "&#%d;", (int)specTempNum);
						strTo.append(buffer, nPrnSize);
					}
					nHighPos = string::npos;
				}
			}
			if(specNum < (unsigned char)' ' || (i == 0 && specNum == ' '))
			{
				nPrnSize = snprintf(buffer, nBufferSize, "&#%d;", (int)specNum);
				strTo.append(buffer, nPrnSize);
			}
			else
			{
				switch (specNum)
				{
				case '>': strTo.append("&gt;", 4); break;
				case '<': strTo.append("&lt;", 4); break;
				case '&': strTo.append("&amp;", 5); break;
				case '\'': strTo.append("&apos;", 6); break;
				case '\"': strTo.append("&quot;", 6); break;
				default: strTo.append(1, specNum); break;
				}
			}
		}
		else
		{
			if(nCharset != ECS_UTF8)
			{
				nPrnSize = snprintf(buffer, nBufferSize, "&#%d;", (int)specNum);
				strTo.append(buffer, nPrnSize);
			}
			else
			{
				bLowByte = false;
				if((specNum >> 3) == 0x1E)
				{
					nLowByteCount = 3;
				}
				else if((specNum >> 4) == 0xE)
				{
					nLowByteCount = 2;
				}
				else if((specNum >> 5) == 0x6)
				{
					nLowByteCount = 1;
				}
				else if((specNum >> 6) == 0x2)
				{
					bLowByte = true;
				}
				else // invalid utf-8 char need to encode
				{
					if(nHighPos != string::npos) // override bytes from last high byte
					{
						sTemp = strTo.substr(nHighPos);
						strTo.erase(nHighPos);
						for(string::size_type j = 0; j < sTemp.size(); j++)
						{
							specTempNum = sTemp[j];
							nPrnSize = snprintf(buffer, nBufferSize, "&#%d;", (int)specTempNum);
							strTo.append(buffer, nPrnSize);
						}
						nHighPos = string::npos;
					}
					nPrnSize = snprintf(buffer, nBufferSize, "&#%d;", (int)specNum);
					strTo.append(buffer, nPrnSize);
					continue;
				}
				if(bLowByte)
				{
					if(nLowByteCount == 0) // missing high byte need encode char
					{
						nPrnSize = snprintf(buffer, nBufferSize, "&#%d;", (int)specNum);
						strTo.append(buffer, nPrnSize);
					}
					else
					{
						if(nLowByteCount == 1)// end of multi byte char, reset high byte position
							nHighPos = string::npos;
						strTo.append(1, specNum);
						nLowByteCount--;
					}
				}
				else
				{
					if(nHighPos != string::npos) // override bites from last height bite and set new height bite pos
					{
						sTemp = strTo.substr(nHighPos);
						strTo.erase(nHighPos);
						for(string::size_type j = 0; j < sTemp.size(); j++)
						{
							specTempNum = sTemp[j];
							nPrnSize = snprintf(buffer, nBufferSize, "&#%d;", (int)specTempNum);
							strTo.append(buffer, nPrnSize);
						}
					}
					strTo.append(1, specNum);
					nHighPos = strTo.size() -1;
				}
			}
		}
	}

	if(!strTo.empty() && nCharset == ECS_UTF8 && nHighPos != string::npos)
	{
		sTemp = strTo.substr(nHighPos);
		strTo.erase(nHighPos);
		for(string::size_type j = 0; j < sTemp.size(); j++)
		{
			specTempNum = sTemp[j];
			nPrnSize = snprintf(buffer, nBufferSize, "&#%d;", (int)specTempNum);
			strTo.append(buffer, nPrnSize);
		}
	}

//	//forward - encode all char <= 32 until other symbol. 
//	for (string::size_type i = 0; i < strFrom.size(); ++i)
//	{
//		unsigned char specNum = strFrom[i];
//		if (((nCharset == ECS_NONE || ECS_ASCII == nCharset) && (specNum < (unsigned char)' ' || specNum >= (unsigned char)'\x80')) || i == 0 && specNum == ' ') 
//		{
//			char buffer[32];
////			sprintf( buffer, "%d", (int)specNum );
////			strTo.append("&#", 2).
////				append(buffer).
////				append(1, ';');
//			sprintf(buffer, "&#%d;", (int)specNum);
//			strTo.append(buffer);
//
//		}
//		else
//		{
//			switch (specNum)
//			{
//			case '>': strTo.append("&gt;", 4); break;
//			case '<': strTo.append("&lt;", 4); break;
//			case '&': strTo.append("&amp;", 5); break;
//			case '\'': strTo.append("&apos;", 6); break;
//			case '\"': strTo.append("&quot;", 6); break;
//			default: strTo.append(1, specNum); break;
//			}
//		}
//	}

	if (!strTo.empty())
	{
		unsigned char specNum = strTo[strTo.size() - 1];
		if (specNum == ' ')
		{
			strTo.resize(strTo.size() - 1);
			strTo.append("&#32;", 5);
		}
	}

	return RESULT_OK;
}

const char CXMLParser::m_chCharacterKinds[0x100] =
{
	/* 0x00   */	0,
	/* 0x01 . */	0,
	/* 0x02 . */	0,
	/* 0x03 . */	0,
	/* 0x04 . */	0,
	/* 0x05 . */	0,
	/* 0x06 . */	0,
	/* 0x07 . */	0,
	/* 0x08 . */	0,
	/* 0x09   */	0,
	/* 0x0A   */	0,
	/* 0x0B  */	0,
	/* 0x0C  */	0,
	/* 0x0D   */	0,
	/* 0x0E . */	0,
	/* 0x0F ¤ */	0,
	/* 0x10 . */	0,
	/* 0x11 . */	0,
	/* 0x12 . */	0,
	/* 0x13 . */	0,
	/* 0x14 ¶ */	0,
	/* 0x15 § */	0,
	/* 0x16 . */	0,
	/* 0x17 . */	0,
	/* 0x18 . */	0,
	/* 0x19 . */	0,
	/* 0x1A . */	0,
	/* 0x1B . */	0,
	/* 0x1C . */	0,
	/* 0x1D . */	0,
	/* 0x1E . */	0,
	/* 0x1F . */	0,
	/* 0x20   */	0,
	/* 0x21 ! */	0,
	/* 0x22 " */	0,
	/* 0x23 # */	0,
	/* 0x24 $ */	0,
	/* 0x25 % */	0,
	/* 0x26 & */	0,
	/* 0x27 ' */	0,
	/* 0x28 ( */	0,
	/* 0x29 ) */	0,
	/* 0x2A * */	0,
	/* 0x2B + */	0,
	/* 0x2C , */	0,
	/* 0x2D - */	BIT_NAME,
	/* 0x2E . */	BIT_NAME,
	/* 0x2F / */	0,
	/* 0x30 0 */	BIT_NAME,
	/* 0x31 1 */	BIT_NAME,
	/* 0x32 2 */	BIT_NAME,
	/* 0x33 3 */	BIT_NAME,
	/* 0x34 4 */	BIT_NAME,
	/* 0x35 5 */	BIT_NAME,
	/* 0x36 6 */	BIT_NAME,
	/* 0x37 7 */	BIT_NAME,
	/* 0x38 8 */	BIT_NAME,
	/* 0x39 9 */	BIT_NAME,
	/* 0x3A : */	BIT_NAME_FIRST | BIT_NAME,
	/* 0x3B ; */	0,
	/* 0x3C < */	0,
	/* 0x3D = */	0,
	/* 0x3E > */	0,
	/* 0x3F ? */	0,
	/* 0x40 @ */	0,
	/* 0x41 A */	BIT_NAME_FIRST | BIT_NAME,
	/* 0x42 B */	BIT_NAME_FIRST | BIT_NAME,
	/* 0x43 C */	BIT_NAME_FIRST | BIT_NAME,
	/* 0x44 D */	BIT_NAME_FIRST | BIT_NAME,
	/* 0x45 E */	BIT_NAME_FIRST | BIT_NAME,
	/* 0x46 F */	BIT_NAME_FIRST | BIT_NAME,
	/* 0x47 G */	BIT_NAME_FIRST | BIT_NAME,
	/* 0x48 H */	BIT_NAME_FIRST | BIT_NAME,
	/* 0x49 I */	BIT_NAME_FIRST | BIT_NAME,
	/* 0x4A J */	BIT_NAME_FIRST | BIT_NAME,
	/* 0x4B K */	BIT_NAME_FIRST | BIT_NAME,
	/* 0x4C L */	BIT_NAME_FIRST | BIT_NAME,
	/* 0x4D M */	BIT_NAME_FIRST | BIT_NAME,
	/* 0x4E N */	BIT_NAME_FIRST | BIT_NAME,
	/* 0x4F O */	BIT_NAME_FIRST | BIT_NAME,
	/* 0x50 P */	BIT_NAME_FIRST | BIT_NAME,
	/* 0x51 Q */	BIT_NAME_FIRST | BIT_NAME,
	/* 0x52 R */	BIT_NAME_FIRST | BIT_NAME,
	/* 0x53 S */	BIT_NAME_FIRST | BIT_NAME,
	/* 0x54 T */	BIT_NAME_FIRST | BIT_NAME,
	/* 0x55 U */	BIT_NAME_FIRST | BIT_NAME,
	/* 0x56 V */	BIT_NAME_FIRST | BIT_NAME,
	/* 0x57 W */	BIT_NAME_FIRST | BIT_NAME,
	/* 0x58 X */	BIT_NAME_FIRST | BIT_NAME,
	/* 0x59 Y */	BIT_NAME_FIRST | BIT_NAME,
	/* 0x5A Z */	BIT_NAME_FIRST | BIT_NAME,
	/* 0x5B [ */	0,
	/* 0x5C \ */	0,
	/* 0x5D ] */	0,
	/* 0x5E ^ */	0,
	/* 0x5F _ */	BIT_NAME_FIRST | BIT_NAME,
	/* 0x60 ` */	0,
	/* 0x61 a */	BIT_NAME_FIRST | BIT_NAME,
	/* 0x62 b */	BIT_NAME_FIRST | BIT_NAME,
	/* 0x63 c */	BIT_NAME_FIRST | BIT_NAME,
	/* 0x64 d */	BIT_NAME_FIRST | BIT_NAME,
	/* 0x65 e */	BIT_NAME_FIRST | BIT_NAME,
	/* 0x66 f */	BIT_NAME_FIRST | BIT_NAME,
	/* 0x67 g */	BIT_NAME_FIRST | BIT_NAME,
	/* 0x68 h */	BIT_NAME_FIRST | BIT_NAME,
	/* 0x69 i */	BIT_NAME_FIRST | BIT_NAME,
	/* 0x6A j */	BIT_NAME_FIRST | BIT_NAME,
	/* 0x6B k */	BIT_NAME_FIRST | BIT_NAME,
	/* 0x6C l */	BIT_NAME_FIRST | BIT_NAME,
	/* 0x6D m */	BIT_NAME_FIRST | BIT_NAME,
	/* 0x6E n */	BIT_NAME_FIRST | BIT_NAME,
	/* 0x6F o */	BIT_NAME_FIRST | BIT_NAME,
	/* 0x70 p */	BIT_NAME_FIRST | BIT_NAME,
	/* 0x71 q */	BIT_NAME_FIRST | BIT_NAME,
	/* 0x72 r */	BIT_NAME_FIRST | BIT_NAME,
	/* 0x73 s */	BIT_NAME_FIRST | BIT_NAME,
	/* 0x74 t */	BIT_NAME_FIRST | BIT_NAME,
	/* 0x75 u */	BIT_NAME_FIRST | BIT_NAME,
	/* 0x76 v */	BIT_NAME_FIRST | BIT_NAME,
	/* 0x77 w */	BIT_NAME_FIRST | BIT_NAME,
	/* 0x78 x */	BIT_NAME_FIRST | BIT_NAME,
	/* 0x79 y */	BIT_NAME_FIRST | BIT_NAME,
	/* 0x7A z */	BIT_NAME_FIRST | BIT_NAME,
	/* 0x7B { */	0,
	/* 0x7C | */	0,
	/* 0x7D } */	0,
	/* 0x7E ~ */	0,
	/* 0x7F  */	0,
	/* 0x80 À */	0,
	/* 0x81 Á */	0,
	/* 0x82 Â */	0,
	/* 0x83 Ã */	0,
	/* 0x84 Ä */	0,
	/* 0x85 Å */	0,
	/* 0x86 Æ */	0,
	/* 0x87 Ç */	0,
	/* 0x88 È */	0,
	/* 0x89 É */	0,
	/* 0x8A Ê */	0,
	/* 0x8B Ë */	0,
	/* 0x8C Ì */	0,
	/* 0x8D Í */	0,
	/* 0x8E Î */	0,
	/* 0x8F Ï */	0,
	/* 0x90 Ð */	0,
	/* 0x91 Ñ */	0,
	/* 0x92 Ò */	0,
	/* 0x93 Ó */	0,
	/* 0x94 Ô */	0,
	/* 0x95 Õ */	0,
	/* 0x96 Ö */	0,
	/* 0x97 × */	0,
	/* 0x98 Ø */	0,
	/* 0x99 Ù */	0,
	/* 0x9A Ú */	0,
	/* 0x9B Û */	0,
	/* 0x9C Ü */	0,
	/* 0x9D Ý */	0,
	/* 0x9E Þ */	0,
	/* 0x9F ß */	0,
	/* 0xA0 à */	0,
	/* 0xA1 á */	0,
	/* 0xA2 â */	0,
	/* 0xA3 ã */	0,
	/* 0xA4 ä */	0,
	/* 0xA5 å */	0,
	/* 0xA6 æ */	0,
	/* 0xA7 ç */	0,
	/* 0xA8 è */	0,
	/* 0xA9 é */	0,
	/* 0xAA ê */	0,
	/* 0xAB ë */	0,
	/* 0xAC ì */	0,
	/* 0xAD í */	0,
	/* 0xAE î */	0,
	/* 0xAF ï */	0,
	/* 0xB0 - */	0,
	/* 0xB1 - */	0,
	/* 0xB2 - */	0,
	/* 0xB3 ¦ */	0,
	/* 0xB4 + */	0,
	/* 0xB5 ¦ */	0,
	/* 0xB6 ¦ */	0,
	/* 0xB7 ¬ */	0,
	/* 0xB8 ¬ */	0,
	/* 0xB9 ¦ */	0,
	/* 0xBA ¦ */	0,
	/* 0xBB ¬ */	0,
	/* 0xBC - */	0,
	/* 0xBD - */	0,
	/* 0xBE - */	0,
	/* 0xBF ¬ */	0,
	/* 0xC0 L */	0,
	/* 0xC1 + */	0,
	/* 0xC2 T */	0,
	/* 0xC3 + */	0,
	/* 0xC4 - */	0,
	/* 0xC5 + */	0,
	/* 0xC6 ¦ */	0,
	/* 0xC7 ¦ */	0,
	/* 0xC8 L */	0,
	/* 0xC9 ã */	0,
	/* 0xCA ¦ */	0,
	/* 0xCB T */	0,
	/* 0xCC ¦ */	0,
	/* 0xCD = */	0,
	/* 0xCE + */	0,
	/* 0xCF ¦ */	0,
	/* 0xD0 ¦ */	0,
	/* 0xD1 T */	0,
	/* 0xD2 T */	0,
	/* 0xD3 L */	0,
	/* 0xD4 L */	0,
	/* 0xD5 - */	0,
	/* 0xD6 ã */	0,
	/* 0xD7 + */	0,
	/* 0xD8 + */	0,
	/* 0xD9 - */	0,
	/* 0xDA - */	0,
	/* 0xDB - */	0,
	/* 0xDC - */	0,
	/* 0xDD ¦ */	0,
	/* 0xDE ¦ */	0,
	/* 0xDF - */	0,
	/* 0xE0 ð */	0,
	/* 0xE1 ñ */	0,
	/* 0xE2 ò */	0,
	/* 0xE3 ó */	0,
	/* 0xE4 ô */	0,
	/* 0xE5 õ */	0,
	/* 0xE6 ö */	0,
	/* 0xE7 ÷ */	0,
	/* 0xE8 ø */	0,
	/* 0xE9 ù */	0,
	/* 0xEA ú */	0,
	/* 0xEB û */	0,
	/* 0xEC ü */	0,
	/* 0xED ý */	0,
	/* 0xEE þ */	0,
	/* 0xEF ÿ */	0,
	/* 0xF0 ¨ */	0,
	/* 0xF1 ¸ */	0,
	/* 0xF2 ª */	0,
	/* 0xF3 º */	0,
	/* 0xF4 ¯ */	0,
	/* 0xF5 ¿ */	0,
	/* 0xF6 ¡ */	0,
	/* 0xF7 ¢ */	0,
	/* 0xF8 ° */	0,
	/* 0xF9 • */	0,
	/* 0xFA · */	0,
	/* 0xFB v */	0,
	/* 0xFC ¹ */	0,
	/* 0xFD ¤ */	0,
	/* 0xFE ¦ */	0,
	/* 0xFF   */	0,
};
