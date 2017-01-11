#ifndef __H_XLSXDOCUMENT__
#define __H_XLSXDOCUMENT__

#pragma region Import the type libraries

#import "libid:2DF8D04C-5BFA-101B-BDE5-00AA0044DE52" \
    rename("RGB", "MSORGB") \
    rename("DocumentProperties", "MSODocumentProperties")

#import "libid:0002E157-0000-0000-C000-000000000046"

#import "libid:00020813-0000-0000-C000-000000000046" \
    rename("DialogBox", "ExcelDialogBox") \
    rename("RGB", "ExcelRGB") \
    rename("CopyFile", "ExcelCopyFile") \
    rename("ReplaceText", "ExcelReplaceText") \
    no_auto_exclude

using namespace Office;
using namespace VBIDE;

#pragma endregion


typedef Excel::Application		XLApplication;
typedef Excel::_ApplicationPtr	XLApplicationPtr;

typedef Excel::WorkbooksPtr		XLWorkBooksPtr;
typedef Excel::_WorkbookPtr		XLWorkBookPtr;

typedef Excel::_WorksheetPtr	XLWorkSheetPrt;

typedef Excel::RangePtr			XLRange;

//////////////////////////////////////////////////
// CWorkSheet declaration

class CWorkSheet
{
public:
	CWorkSheet( XLWorkSheetPrt xlwsWorkSheet );

	void SetName( const wstring& sSheetName );

	bool SetRangeValues( const wstring_vv& vvsValues, const wstring& wsRange ) const;

protected:
	bool SafeArrayPutValues( SAFEARRAY* psa, long index, const wstring_v& vsValues ) const;

private:
	XLWorkSheetPrt		m_xlwsWorkSheet;
};

//////////////////////////////////////////////////
// CWorkBook declaration

class CWorkBook
{
public:
	CWorkBook( XLWorkBookPtr pxlwbWorkBook );

	CWorkSheet GetActiveSheet() const;

	bool SaveToFile( const string& sFileName ) const;
	bool Close() const;
	bool SaveAndClose( const string& sFileName ) const;

private:
	XLWorkBookPtr	m_pxlwbWorkBook;
};

//////////////////////////////////////////////////
// CXLSXDocument declaration

class CExcelApp
{
public:
	CExcelApp();
	~CExcelApp();

	bool GetIsInitialized() const { return m_bInitialized; }

	void MakeVisible( bool bVisible = true ) const;
	CWorkBook CreateWorkBook() const;

protected:
	bool Initialize();
	void Quit();

private:
	bool			m_bInitialized;

	XLApplicationPtr	m_XLApplicationPtr;
};

#endif // __H_XLSXDOCUMENT__
