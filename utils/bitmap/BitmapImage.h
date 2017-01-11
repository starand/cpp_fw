#ifndef __BMPFILE_H_INCLUDED
#define __BMPFILE_H_INCLUDED

class CRGBTriple;

//////////////////////////////////////////////////
// CBMPFile declaration

class CBitmapImage
{
public:
	CBitmapImage();
	CBitmapImage(const CBitmapImage &biImage);
	~CBitmapImage();

private:
#pragma pack (1)
	typedef struct
	{
		ushort	m_usType;
		uint	m_uiFileSize;
		ushort	m_usReserved1;
		ushort	m_usReserved2;
		uint	m_uiDataOffset;
	} CBitmapFileHeader;
#pragma pack ()

	typedef struct  
	{
		uint	m_uiHeaderSize;
		uint	m_uiImageWidth;
		uint	m_uiImageHeight;
		ushort	m_usFlatsCount;
		ushort	m_usBitsPerPixel;
		uint	m_uiCompressionType;
		uint	m_uiCompressedSize;
		uint	m_uiHorizontalPM;
		uint	m_uiVerticalPM;
		uint	m_uiColorsCount;
		uint	m_uiImportantColorsCount;
	} CBitmapInfoHeader;

	enum ECOMPRESSIONTYPE
	{
		ECT_NOTCOMPRESSION,
		ECT_RLE8,
		ECT_RLE4,
	};

public:
	bool LoadFromFile(const string &sFileName);
	bool SaveToFile(const string &sFileName) const;

	void CloneImage(const CBitmapImage &biImage, bool bCopyData = true);

public:
	sizeint GetHeight() const;
	sizeint GetWidth() const;

	sizeint GetDataSize() const { return GetPixelsDataSize(); }
	void *GetImageData() const { return GetPixelsData(); }

	uchar GetBitDepth() const;
	uchar GetBytesPerPixel() const;

	CRGBTriple &GetPixel(sizeint siYPos, sizeint siXPos) const;
	bool GetPixel(sizeint siYPos, sizeint siXPos, CRGBTriple &rgbOutPixel) const;

	bool SetPixel(sizeint siYPos, sizeint siXPos, const CRGBTriple &rgbOutPixel) const;

private:
	sizeint GetDataOffset() const;

public:
	sizeint GetLineSize() const { return ALIGN_UP(GetWidth() * GetBytesPerPixel(), 4); }

private:
	sizeint GetLineOffset(sizeint nLineNumber) const;

	sizeint GetPixelOffset(sizeint siPixelIndex) const;

private:
	void DestroyInternalObjects();

	void AllocatePixelsData(sizeint siPixelsDataSize);
	void FreePixelsData();

private:
	CBitmapFileHeader &GetBitmapFileHeader() const { return const_cast<CBitmapFileHeader &>(m_fhBitmapFileHeader); }
	CBitmapInfoHeader &GetBitmapInfoHeader() const { return const_cast<CBitmapInfoHeader &>(m_biBitmapInfoHeader); }

	void *GetPixelsData() const { return m_pcPixelsData; }
	void SetPixelsData(void *pvValue) { m_pcPixelsData = pvValue;}

	sizeint GetPixelsDataSize() const { return m_siPixelsDataSize; }
	void SetPixelsDataSize(sizeint siValue) { m_siPixelsDataSize = siValue; }

private:
	CBitmapFileHeader	m_fhBitmapFileHeader;
	CBitmapInfoHeader	m_biBitmapInfoHeader;

	void				*m_pcPixelsData;
	sizeint				m_siPixelsDataSize;
};

#endif // __BMPFILE_H_INCLUDED
