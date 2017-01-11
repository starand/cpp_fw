#include "stdafx.h"
#include "display.h"
#include <WinUser.h>
//#include <Windows.h>
#include <Olectl.h>


bool display_t::bDisplayIsOff = false;


//////////////////////////////////////////////////
// CDisplay implementation

/*static */
void display_t::turn_off( size_t nTimeOut /*= USER_INTERACTION_TIMEOUT*/ )
{
	Sleep( nTimeOut );
	SendMessage( HWND_BROADCAST, WM_SYSCOMMAND, SC_MONITORPOWER, (LPARAM)2 );
	bDisplayIsOff = true;
}

/*static */
void display_t::turn_on()
{
	SendMessage( HWND_BROADCAST, WM_SYSCOMMAND, SC_MONITORPOWER, (LPARAM)-1 );
	bDisplayIsOff = false;
}


/*static */
void display_t::change_state()
{
	if (is_off())
	{
		turn_off();
	}
	else
	{
		turn_on();
	}
}

/*static */
bool display_t::save_bitmap(const char *szFileName, HBITMAP bmp, HPALETTE pal /*= NULL*/)
{
	START_FUNCTION_BOOL();

	PICTDESC pdPictDesc;
	pdPictDesc.cbSizeofstruct = sizeof(PICTDESC);
	pdPictDesc.picType = PICTYPE_BITMAP;
	pdPictDesc.bmp.hbitmap = bmp;
	pdPictDesc.bmp.hpal = pal;

	LPPICTURE plPicture = NULL;
	HRESULT hResult = OleCreatePictureIndirect(&pdPictDesc, IID_IPicture, false, reinterpret_cast<void**>(&plPicture));

	if (!SUCCEEDED(hResult))
	{
		break;
	}

	LPSTREAM lpStream = NULL;
	hResult = CreateStreamOnHGlobal(0, true, &lpStream);

	if (!SUCCEEDED(hResult))
	{
		plPicture->Release();
		return false;
	}

	LONG lBytesStreamed = 0;
	hResult = plPicture->SaveAsFile(lpStream, true, &lBytesStreamed);

	HANDLE hFile = CreateFileA(szFileName, GENERIC_WRITE, FILE_SHARE_READ, 0,
		CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, 0);

	if (!SUCCEEDED(hResult) || !hFile)
	{
		lpStream->Release();
		plPicture->Release();
		break;
	}

	HGLOBAL hgMemory = 0;
	GetHGlobalFromStream(lpStream, &hgMemory);
	LPVOID lpData = GlobalLock(hgMemory);

	DWORD dwBytesWritten = 0;

	bool bResult = !!WriteFile(hFile, lpData, lBytesStreamed, &dwBytesWritten, 0);
	bResult &= (dwBytesWritten == static_cast<DWORD>(lBytesStreamed));

	GlobalUnlock(hgMemory);
	CloseHandle(hFile);

	lpStream->Release();
	plPicture->Release();

	END_FUNCTION_BOOL();
}

/*static */
void display_t::save_screen(const char *szFileName)
{
	ASSERTE(szFileName);

	int nScreenWidth = GetSystemMetrics(SM_CXSCREEN);
	int nScreenHeight = GetSystemMetrics(SM_CYSCREEN);

	HWND hDesktopWnd = GetDesktopWindow();
	HDC hDesktopDC = GetDC(hDesktopWnd);
	HDC hCaptureDC = CreateCompatibleDC(hDesktopDC);

	HBITMAP hCaptureBitmap = CreateCompatibleBitmap(hDesktopDC, nScreenWidth, nScreenHeight);

	SelectObject(hCaptureDC, hCaptureBitmap); 
	BitBlt(hCaptureDC, 0, 0, nScreenWidth, nScreenHeight, hDesktopDC, 0, 0, SRCCOPY | CAPTUREBLT);

	save_bitmap(szFileName, hCaptureBitmap);

	ReleaseDC(hDesktopWnd,hDesktopDC);
	DeleteDC(hCaptureDC);
	DeleteObject(hCaptureBitmap);
}
