#ifndef _ELXMLPARSE_
#define _ELXMLPARSE_

class XMLDOM_Document;
class XMLDOM_Node;
class XMLDOM_Element;

enum ParseResult
{
	RESULT_CLOSE_TAG = -2,
	RESULT_OUT_OF_DATA = -2, // same meaning as RESULT_CLOSE_TAG
	RESULT_SKIPPED = -1,
	RESULT_OK = 0,
	RESULT_ONLY_ONE_ROOT_ELEMENT_ALLOWED,
	RESULT_INVALID_NODE,
	RESULT_CLOSE_TAG_WO_START_TAG,
	RESULT_START_TAG_WO_CLOSE_TAG,
	RESULT_ERROR_PARSING_ATTRS,
	RESULT_ERROR_PARSING_TAG_HEADER,
	RESULT_TAG_NAME_INVALID,
	RESULT_DOC_DEFINITON_EXPECTED,
	RESULT_SPECIFIED_ATTRIBUTE_UNEXPECTED_AT_POS,
	RESULT_INVALID_SYNTAX_FOR_XML_DECL,
	RESULT_TOP_LEVEL_ELEMENT_REQUIRED,
	RESULT_SEMICOLON_EXPECTED,
	RESULT_UNEXPECTED_END_OF_FILE,
	RESULT_WHITESPACE_NOT_ALLOWED_AT_POS,
	RESULT_E_NO_IMPL,
	RESULT_WHITESPACE_EXPECTED_AT_POS,
	RESULT_CDATA_SECTION_WAS_NOT_CLOSED,
	RESULT_COMMENT_WAS_NOT_CLOSED,
	RESULT_INVALID_STRING,
	RESULT_TAG_WAS_NOT_CLOSED,
	RESULT_INTERNAL_ERROR,
	RESULT_NOT_SUPPORTED_CHARSET
};

class CXMLParserStringCache
{
public:
	void TranslateString(const char *szBegin, const char *szEnd, string &sResult);

private:
//	string					m_asCache[0x256];
	string					m_asCache[256];
};

class CXMLParser 
{
public:
	CXMLParser();
	~CXMLParser();
	
	int ParseDocument(const string &sXMLText, void *fpFilter, void *pvContext, XMLDOM_Document *pdDocument);
	int StringToNode(const string &sXMLText, XMLDOM_Document *pdDocument, XMLDOM_Node *&pnNode);
	
	const char *ExplainLastError() { return ExplainError(GetLastError(), m_chMessageBuffer, GetErrorPos()); }
	int GetLastError() const { return m_iParseResult; }
	unsigned int GetErrorPos() const { return m_iParseResult != RESULT_OK ? m_uErrorPosition : (unsigned int)0 - (unsigned int)1; }

private:
	struct CParseContext
	{
		CParseContext(XMLDOM_Document *pdDocument, void *fpFilter, void *pvContext, unsigned int nLevel):
			m_pdDocument(pdDocument), m_fpFilter(fpFilter), m_pvContext(pvContext), m_nLevel(nLevel) {}

		XMLDOM_Document *m_pdDocument;
		void			*m_fpFilter;
		void			*m_pvContext;
		unsigned int	m_nLevel;
	};

	int InternalParseDocument(const char *szBegin, const char *szEnd, const char *&szResult, CParseContext &pcContext);
	int ParseDocDefinition(const char *szBegin, const char *szEnd, const char *&szResult, CParseContext &pcContext);
	int RecognizeAndBuildNode(const char *szBegin, const char *szEnd, bool bSkipped, const char *&szResult, CParseContext &pcContext, XMLDOM_Node *&pnOutNode);
	int BuildElement(const char *szBegin, const char *szEnd, bool bSkipped, const char *&szResult, CParseContext &pcContext, XMLDOM_Element *&peOutElement);
	int ParseTagHeader(const char *szBegin, const char *szEnd, bool bSkipped, const char *&szResult, 
		string &sTagName, string &sTagNamespace, vector<string> &vsAttrNames, vector<string> &vsAttrValues, bool &bIsEmpty);
	int ParseCloseTagHeader(const char *szBegin, const char *szEnd, bool bSkipped, const char *&szResult, 
		string &sTagName, string &sTagNamespace);
	int ParseTagHeaderName(const char *szBegin, const char *szEnd, bool bSkipped, const char *&szResult, 
		string &sTagName, string &sTagNamespace);
	int ParseTagHeaderAttributes(const char *szBegin, const char *szEnd, bool bSkipped, const char *&szResult, 
		vector<string> &vsAttrNames, vector<string> &vsAttrValues);
	int ParseAttribute(const char *szBegin, const char *szEnd, bool bSkipped, const char *&szResult, 
		string &sAttrName, string &sAttrValue);
	static int ParseComment(const char *szBegin, const char *szEnd, bool bSkipped, const char *&szResult, string &sData);
	static int ParseCDATASection(const char *szBegin, const char *szEnd, bool bSkipped, const char *&szResult, string &sData);
	static int ParseTextFragment(const char *szBegin, const char *szEnd, bool bSkipped, const char *&szResult, string &sData);
	static int ConvertSymbolFromSpec(const char *szBegin, const char *szEnd, const char *&szResult, char &chConverted);

protected:
	static void StripSpaces(const char *szBegin, const char *szEnd, const char *&szNewBegin);

protected:
	static bool IsStartOfDocDefinition(const char *szBegin, const char *szEnd, const char *&szNewBegin);
	static bool IsEndOfDocDefinition (const char *szBegin, const char *szEnd, const char *&szNewBegin);
	static bool IsLT(const char *szBegin, const char *szEnd, const char *&szNewBegin);
	static bool IsGT(const char *szBegin, const char *szEnd, const char *&szNewBegin);
	static bool IsCloseTag(const char *szBegin, const char *szEnd, const char *&szNewBegin);
	static bool IsEmptyTagGT(const char *szBegin, const char *szEnd, const char *&szNewBegin);
	static bool IsAmpLt(const char *szBegin, const char *szEnd, const char *&szNewBegin);
	static bool IsAmpGt(const char *szBegin, const char *szEnd, const char *&szNewBegin);
	static bool IsAmpAmp(const char *szBegin, const char *szEnd, const char *&szNewBegin);
	static bool IsAmpApos(const char *szBegin, const char *szEnd, const char *&szNewBegin);
	static bool IsAmpQuot(const char *szBegin, const char *szEnd, const char *&szNewBegin);
	static bool IsStartOfComment(const char *szBegin, const char *szEnd, const char *&szNewBegin);
	static bool SeekEndOfComment(const char *szBegin, const char *szEnd, const char *&szCommentEnd, const char *&szNewBegin);
	static bool IsStartOfCDATASection(const char *szBegin, const char *szEnd, const char *&szNewBegin);
	static bool SeekEndOfCDATASection(const char *szBegin, const char *szEnd, const char *&szCDATAEnd, const char *&szNewBegin);
	static bool IsTextFragment(const char *szBegin, const char *szEnd);
	
protected:
	static inline bool IsValidXMLNameFirtsChar(char ch);
	static inline bool IsValidXMLNameChar(char ch);
	static inline bool IsTextTerminator(char ch);
	static inline bool IsAmpPrefix(char ch);
	static inline bool IsAmpASCIIPrefix(char ch);
	static inline bool IsAmpASCIIHexPrefix(char ch);
	static inline bool IsAmpTerminator(char ch);
	static inline bool IsQuote(char ch);
	static inline bool IsAttrValueDelimiter(char ch);
	static inline bool IsSpace(unsigned char ch);

protected:
	static const char *ExplainError(int iParseResult, char *chMessageBuffer, unsigned int uErrorPosition);
	
private:
	CXMLParserStringCache	m_scStringCache;
	unsigned int			m_uErrorPosition;
	int						m_iParseResult;
	char					m_chMessageBuffer[200];

private:
	static const char		m_chCharacterKinds[0x100];
};



int ConvertStringToSpec (const string &strFrom, string &strTo, EncodingCharset nCharset);

#endif // _ELXMLPARSE_
