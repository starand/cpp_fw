#include "StdAfx.h"

#include "BitmapImage.h"
#include "BitmapPixel.h"

#include "fileutils.h"
#include "consts.h"


#define BYTE_SIZE	8

//////////////////////////////////////////////////
// CBMPFile implementation

CBitmapImage::CBitmapImage():
	m_fhBitmapFileHeader(),
	m_biBitmapInfoHeader(),
	m_pcPixelsData(NULL),
	m_siPixelsDataSize(0)
{
	static_assert(sizeof(CBitmapFileHeader) == 14, "Invalid CBitmapFileHeader structure size");
	static_assert(sizeof(CBitmapInfoHeader) == 40, "Invalid CBitmapInfoHeader structure size");
}

CBitmapImage::CBitmapImage(const CBitmapImage &biImage):
	m_fhBitmapFileHeader(),
	m_biBitmapInfoHeader(),
	m_pcPixelsData(NULL),
	m_siPixelsDataSize(0)
{
	static_assert(sizeof(CBitmapFileHeader) == 14, "Invalid CBitmapFileHeader structure size");
	static_assert(sizeof(CBitmapInfoHeader) == 40, "Invalid CBitmapInfoHeader structure size");

	CloneImage(biImage);
}

CBitmapImage::~CBitmapImage()
{
	DestroyInternalObjects();
}


bool CBitmapImage::LoadFromFile(const string &sFileName)
{
	START_FUNCTION_BOOL();

	FileUtils::CFileWrapper fpBMPFile(sFileName.c_str(), "rb");

	if (!fpBMPFile)
	{
		LOG_ERROR2_BREAK(g_szUnableToOpenFile, sFileName.c_str());
	}

	CBitmapFileHeader &fhBitmapHeaderHeader = GetBitmapFileHeader();

	if (sizeof(fhBitmapHeaderHeader) != fread(&fhBitmapHeaderHeader, 1, sizeof(fhBitmapHeaderHeader), fpBMPFile))
	{
		LOG_ERROR2_BREAK(g_szUnableToReadFile, sFileName.c_str());
	}

	CBitmapInfoHeader &biBitmapInfoHeader = GetBitmapInfoHeader();

	if (sizeof(biBitmapInfoHeader) != fread(&biBitmapInfoHeader, 1, sizeof(biBitmapInfoHeader), fpBMPFile))
	{
		LOG_ERROR2_BREAK(g_szUnableToReadFile, sFileName.c_str());
	}

	sizeint siDataSize = fhBitmapHeaderHeader.m_uiFileSize - sizeof(CBitmapFileHeader) - sizeof(CBitmapInfoHeader);

	SetPixelsDataSize(siDataSize);
	AllocatePixelsData(siDataSize);

	void *pvPixelsData = GetPixelsData();
	if (siDataSize != fread(pvPixelsData, 1, siDataSize, fpBMPFile))
	{
		LOG_ERROR2_BREAK("Incorrect image size", sFileName.c_str());
	}

	END_FUNCTION_BOOL();
}

bool CBitmapImage::SaveToFile(const string &sFileName) const
{
	START_FUNCTION_BOOL();

	FileUtils::CFileWrapper fpBMPFile(sFileName.c_str(), "wb");

	if (!fpBMPFile)
	{
		LOG_ERROR2_BREAK(g_szUnableToOpenFile, sFileName.c_str());
	}

	CBitmapFileHeader &fhBitmapHeaderHeader = GetBitmapFileHeader();

	if (sizeof(fhBitmapHeaderHeader) != fwrite(&fhBitmapHeaderHeader, 1, sizeof(fhBitmapHeaderHeader), fpBMPFile))
	{
		LOG_ERROR2_BREAK(g_szUnableToWriteFile, sFileName.c_str());
	}

	CBitmapInfoHeader &biBitmapInfoHeader = GetBitmapInfoHeader();

	if (sizeof(biBitmapInfoHeader) != fwrite(&biBitmapInfoHeader, 1, sizeof(biBitmapInfoHeader), fpBMPFile))
	{
		LOG_ERROR2_BREAK(g_szUnableToWriteFile, sFileName.c_str());
	}

	void *pvPixelsData = GetPixelsData();
	sizeint siPixelsDataSize = GetPixelsDataSize();

	if (siPixelsDataSize != fwrite(pvPixelsData, 1, siPixelsDataSize, fpBMPFile))
	{
		LOG_ERROR2_BREAK(g_szUnableToWriteFile, sFileName.c_str());
	}

	END_FUNCTION_BOOL();
}


void CBitmapImage::CloneImage(const CBitmapImage &biImage, bool bCopyData /*= true*/)
{
	m_fhBitmapFileHeader = biImage.m_fhBitmapFileHeader;
	m_biBitmapInfoHeader = biImage.m_biBitmapInfoHeader;

	FreePixelsData();
	
	m_siPixelsDataSize = biImage.m_siPixelsDataSize;
	AllocatePixelsData(m_siPixelsDataSize);

	if (bCopyData)
	{
		memcpy(m_pcPixelsData, biImage.m_pcPixelsData, m_siPixelsDataSize);
	}
	else
	{
		memset(m_pcPixelsData, 0, m_siPixelsDataSize);
	}
}


sizeint CBitmapImage::GetHeight() const
{
	CBitmapInfoHeader &biBitmapInfoHeader = GetBitmapInfoHeader();
	ASSERTE(biBitmapInfoHeader.m_uiImageHeight);

	sizeint siResult = biBitmapInfoHeader.m_uiImageHeight;
	return siResult;
}

sizeint CBitmapImage::GetWidth() const
{
	CBitmapInfoHeader &biBitmapInfoHeader = GetBitmapInfoHeader();
	ASSERTE(biBitmapInfoHeader.m_uiImageWidth);

	sizeint siResult = biBitmapInfoHeader.m_uiImageWidth;
	return siResult;
}


uchar CBitmapImage::GetBitDepth() const
{
	CBitmapInfoHeader &biBitmapInfoHeader = GetBitmapInfoHeader();
	ASSERTE(biBitmapInfoHeader.m_usBitsPerPixel);

	uchar ucResult = (uchar)biBitmapInfoHeader.m_usBitsPerPixel;
	return ucResult;
}

uchar CBitmapImage::GetBytesPerPixel() const 
{
	uchar ucResult = (GetBitDepth() + BYTE_SIZE - 1) / BYTE_SIZE; 
	return ucResult;
}


CRGBTriple &CBitmapImage::GetPixel(sizeint siYPos, sizeint siXPos) const
{
	CBitmapInfoHeader &biBitmapInfoHeader = GetBitmapInfoHeader();
	ASSERTE(biBitmapInfoHeader.m_uiImageHeight > siYPos && biBitmapInfoHeader.m_uiImageWidth > siXPos);

	void *pvPixelsData = GetPixelsData();
	sizeint siPixelOffset = GetLineOffset(siYPos) +  GetPixelOffset(siXPos);
	
	CRGBTriple *prgbPixel = (CRGBTriple *)(((char*)pvPixelsData) + siPixelOffset);
	return *prgbPixel;
}

bool CBitmapImage::GetPixel(sizeint siYPos, sizeint siXPos, CRGBTriple &rgbOutPixel) const
{
	START_FUNCTION_BOOL();

	CBitmapInfoHeader &biBitmapInfoHeader = GetBitmapInfoHeader();
	ASSERTE(biBitmapInfoHeader.m_uiImageHeight && biBitmapInfoHeader.m_uiImageWidth);

	if (siYPos >= biBitmapInfoHeader.m_uiImageHeight || siXPos >= biBitmapInfoHeader.m_uiImageWidth)
	{
		LOG_ERROR_BREAK("Invalid pixel pos");
	}

	rgbOutPixel = GetPixel(siYPos, siXPos);
	END_FUNCTION_BOOL();
}


bool CBitmapImage::SetPixel(sizeint siYPos, sizeint siXPos, const CRGBTriple &rgbOutPixel) const
{
	START_FUNCTION_BOOL();

	CBitmapInfoHeader &biBitmapInfoHeader = GetBitmapInfoHeader();
	ASSERTE(biBitmapInfoHeader.m_uiImageHeight && biBitmapInfoHeader.m_uiImageWidth);

	if (siYPos >= biBitmapInfoHeader.m_uiImageHeight || siXPos >= biBitmapInfoHeader.m_uiImageWidth)
	{
		LOG_ERROR_BREAK("Invalid pixel pos");
	}

	void *pvPixelsData = GetPixelsData();
	sizeint siPixelOffset = GetLineOffset(siYPos) +  GetPixelOffset(siXPos);

	CRGBTriple *prgbPixel = (CRGBTriple *)(((char*)pvPixelsData) + siPixelOffset);
	*prgbPixel = rgbOutPixel;

	END_FUNCTION_BOOL();
}


sizeint CBitmapImage::GetDataOffset() const
{
	CBitmapFileHeader &fhBitmapHeaderHeader = GetBitmapFileHeader();
	ASSERTE(fhBitmapHeaderHeader.m_uiDataOffset);

	sizeint siResult = fhBitmapHeaderHeader.m_uiDataOffset;
	return siResult;
}


sizeint CBitmapImage::GetLineOffset(sizeint nLineNumber) const
{
	sizeint siBackwardLineNumber = GetHeight() - 1 - nLineNumber;

	sizeint siResult = GetLineSize() * siBackwardLineNumber;
	return siResult;
}


sizeint CBitmapImage::GetPixelOffset(sizeint siPixelIndex) const
{
	sizeint siResult = GetBitDepth() * siPixelIndex / BYTE_SIZE;
	return siResult;
}


void CBitmapImage::DestroyInternalObjects()
{
	FreePixelsData();
}


void CBitmapImage::AllocatePixelsData(sizeint siPixelsDataSize)
{
	if (GetPixelsData() != NULL)
	{
		int x = 0;
	}

	ASSERTE(GetPixelsData() == NULL);

	void *pvPixelsData = malloc(siPixelsDataSize);
	ASSERTE(pvPixelsData);

	SetPixelsData(pvPixelsData);
}

void CBitmapImage::FreePixelsData()
{
	void *pvPixelsData = GetPixelsData();

	if (pvPixelsData)
	{
		SetPixelsData(NULL);
		free(pvPixelsData);
	}
}
