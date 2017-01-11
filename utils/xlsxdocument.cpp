#include "stdafx.h"
#include "xlsxdocument.h"
#include <Windows.h>
#include <combaseapi.h>
#include "consts.h"

//////////////////////////////////////////////////
// CXLSXDocument implementation

CExcelApp::CExcelApp() : m_bInitialized(false)
{
	Initialize();
}

CExcelApp::~CExcelApp()
{
    Quit();
}

bool CExcelApp::Initialize()
{
	START_FUNCTION_BOOL();

	// Initializes the COM library on the current thread and identifies the
	if( S_OK != CoInitializeEx(NULL, COINIT_APARTMENTTHREADED) ) {
		LOG_ERROR_BREAK( "Unable to initialize COM library" );
	}

    HRESULT hrXLAppInstance = m_XLApplicationPtr.CreateInstance( __uuidof(XLApplication) );
    if( FAILED(hrXLAppInstance) ) {
        LOG_ERROR2_BREAK( "Unable to create XLApp Instance 0x%08lx", hrXLAppInstance );
    }

	m_bInitialized = true;
	END_FUNCTION_BOOL();
}

void CExcelApp::Quit()
{
	m_bInitialized = false;
	m_XLApplicationPtr->Quit();
	CoUninitialize(); // Uninitialize COM for this thread.
}

void CExcelApp::MakeVisible( bool bVisible /*= true*/ ) const
{
	m_XLApplicationPtr->Visible[ 0 ] = ( bVisible ? VARIANT_TRUE : VARIANT_FALSE );
}

CWorkBook CExcelApp::CreateWorkBook() const
{
    XLWorkBooksPtr xlwbWorkBooks = m_XLApplicationPtr->Workbooks;
    return CWorkBook( xlwbWorkBooks->Add() );
}

//////////////////////////////////////////////////
// CWorkBook implementation

CWorkBook::CWorkBook( XLWorkBookPtr pxlwbWorkBook ) : m_pxlwbWorkBook(pxlwbWorkBook) 
{
}

CWorkSheet CWorkBook::GetActiveSheet() const
{
	return CWorkSheet( m_pxlwbWorkBook->ActiveSheet );
}

bool CWorkBook::SaveToFile( const string& sFileName ) const
{
	START_FUNCTION_BOOL();

	variant_t vtFileName( sFileName.c_str() );
	HRESULT hResult = m_pxlwbWorkBook->SaveAs( vtFileName, Excel::xlOpenXMLWorkbook, vtMissing, vtMissing, vtMissing, vtMissing, Excel::xlNoChange );
    if( FAILED(hResult) ) LOG_ERROR2_BREAK( "Unable to save workbook 0x%08lx", hResult );

	END_FUNCTION_BOOL();
}

bool CWorkBook::Close() const
{
	START_FUNCTION_BOOL();

	HRESULT hResult = m_pxlwbWorkBook->Close();
	if( FAILED(hResult) ) LOG_ERROR2_BREAK( "Unable to close workbook 0x%08lx", hResult );

	END_FUNCTION_BOOL();
}

bool CWorkBook::SaveAndClose( const string& sFileName ) const
{
	return SaveToFile(sFileName) && Close();
}

//////////////////////////////////////////////////
// CWorkSheet implementation

CWorkSheet::CWorkSheet( XLWorkSheetPrt xlwsWorkSheet ) : m_xlwsWorkSheet(xlwsWorkSheet)
{
}

void CWorkSheet::SetName( const wstring& sSheetName )
{
	m_xlwsWorkSheet->Name = _bstr_t( sSheetName.c_str() );
}

bool CWorkSheet::SetRangeValues( const wstring_vv& vvsValues, const wstring& wsRange ) const
{
	START_FUNCTION_BOOL();

	if( vvsValues.empty() || vvsValues[0].empty() ) LOG_ERROR_BREAK( "Regeon should not be empty" );
	
	const size_t nRowsCount = vvsValues.size();
	const size_t nColsCount = vvsValues[0].size();

    // Construct a 5 x 2 safearray of user names
    VARIANT saNames;
    saNames.vt = VT_ARRAY | VT_VARIANT;
    {
        SAFEARRAYBOUND* pSafeArrayBounds = new (nothrow) SAFEARRAYBOUND[ nColsCount ];
		_ASSERT_MEMORY_BREAK( pSafeArrayBounds );

        pSafeArrayBounds[0].lLbound = 1; 
		pSafeArrayBounds[0].cElements = nRowsCount;
        pSafeArrayBounds[1].lLbound = 1; 
		pSafeArrayBounds[1].cElements = nColsCount;

        saNames.parray = SafeArrayCreate( VT_VARIANT, 2, pSafeArrayBounds ); // 2 - number of dimensions
		for( size_t nRowIdx = 0;  nRowIdx < nRowsCount; ++nRowIdx )
		{
			SafeArrayPutValues(saNames.parray, nRowIdx + 1, vvsValues[nRowIdx] );
		}

		delete [] pSafeArrayBounds;
    }

    // Get Range object for the Range from wsRange
    VARIANT param;
    param.vt = VT_BSTR;
	param.bstrVal = SysAllocString( wsRange.c_str() );
    XLRange xlRange = m_xlwsWorkSheet->Range[ param ];
    xlRange->Value2 = saNames;

    // Clear the safearray
    VariantClear( &saNames );

	END_FUNCTION_BOOL();
}

bool CWorkSheet::SafeArrayPutValues( SAFEARRAY* pSafeArray, long index, const wstring_v& vsValues ) const
{
	START_FUNCTION_BOOL();
    HRESULT hResult = E_FAIL;

	size_t nValuesCount = vsValues.size();
	for( size_t idx = 0; idx < nValuesCount; ++idx )
	{
		long nIdicies[] = { index, idx + 1 };

		VARIANT vtValue;
		vtValue.vt = VT_BSTR;
		vtValue.bstrVal = SysAllocString( vsValues[idx].c_str() );

		// Copies the VARIANT into the SafeArray
		hResult = SafeArrayPutElement( pSafeArray, nIdicies, (void*)&vtValue );
		VariantClear( &vtValue );		

		if( FAILED(hResult) ) break;
	}

	if( FAILED(hResult) ) break;
	END_FUNCTION_BOOL();
}
