#include "StdAfx.h"

#include "BitmapUtils.h"
#include "BitmapImage.h"


namespace BitmapUtils
{

CRGBTriple AvgBlender(sizeint siYPos, sizeint siXPos, const CBitmapImage &bmpImage, sizeint siRadius)
{
	ASSERTE(siXPos < bmpImage.GetWidth() && siYPos < bmpImage.GetHeight());

	sizeint siRed = 0, siGreen = 0, siBlue = 0, siItemsCount = 0;

	sizeint siMinY = _MAX((int)(siYPos - siRadius), 0);
	sizeint siMaxY = _MIN((int)(siYPos + siRadius), bmpImage.GetHeight() - 1);
	sizeint siMinX = _MAX((int)(siXPos - siRadius), 0);
	sizeint siMaxX = _MIN((int)(siXPos + siRadius), bmpImage.GetWidth() - 1);

	for (sizeint siYIndex = siMinY; siYIndex <= siMaxY; ++siYIndex)
	{
		for (sizeint siXIndex = siMinX; siXIndex <= siMaxX; ++siXIndex)
		{
			const CRGBTriple &rgbPixel = bmpImage.GetPixel(siYIndex, siXIndex);

			siRed += rgbPixel.Red();
			siGreen += rgbPixel.Green();
			siBlue += rgbPixel.Blue();

			++siItemsCount;
		}
	}

	siRed /= siItemsCount;
	siGreen /= siItemsCount;
	siBlue /= siItemsCount;

	return CRGBTriple(siRed, siGreen, siBlue);
}

CRGBTriple MedianBlender(sizeint siYPos, sizeint siXPos, const CBitmapImage &bmpImage, sizeint siRadius)
{
	ASSERTE(siXPos < bmpImage.GetWidth() && siYPos < bmpImage.GetHeight());

	sizeint siRed = 0, siGreen = 0, siBlue = 0, siItemsCount = 0;

	sizeint siMinY = _MAX((int)(siYPos - siRadius), 0);
	sizeint siMaxY = _MIN((int)(siYPos + siRadius), bmpImage.GetHeight() - 1);
	sizeint siMinX = _MAX((int)(siXPos - siRadius), 0);
	sizeint siMaxX = _MIN((int)(siXPos + siRadius), bmpImage.GetWidth() - 1);

	std::vector<uchar> vuRed, vuGreen, vuBlue;

	for (sizeint siYIndex = siMinY; siYIndex <= siMaxY; ++siYIndex)
	{
		for (sizeint siXIndex = siMinX; siXIndex <= siMaxX; ++siXIndex)
		{
			const CRGBTriple &rgbPixel = bmpImage.GetPixel(siYIndex, siXIndex);

			vuRed.push_back(rgbPixel.Red());
			vuGreen.push_back(rgbPixel.Green());
			vuBlue.push_back(rgbPixel.Blue());
		}
	}

	ASSERTE(vuRed.size());

	std::sort(vuRed.begin(), vuRed.end());
	std::sort(vuGreen.begin(), vuGreen.end());
	std::sort(vuBlue.begin(), vuBlue.end());

	sizeint siMidIdx = vuRed.size() / 2;
	return CRGBTriple(vuRed[siMidIdx], vuGreen[siMidIdx], vuBlue[siMidIdx]);
}


void BlendImagePixels(const CBitmapImage &biInputImage, CBitmapImage &biOutImage, PixelBlender fPixelBlender, sizeint siRadius /*= 1*/, sizeint siTimes /*= 1*/)
{
	biOutImage.CloneImage(biInputImage);

	sizeint siImageWidth = biInputImage.GetWidth();
	sizeint siImageHeight = biInputImage.GetHeight();
	
	for (sizeint i = 0; i < siTimes; ++i)
	{
		CBitmapImage biInputCopy(biOutImage);
		for (sizeint siYPos = 0; siYPos < siImageHeight; ++siYPos)
		{
			for (sizeint siXPos = 0; siXPos < siImageWidth; ++siXPos)
			{
				CRGBTriple &&rgbTripple = fPixelBlender(siYPos, siXPos, biInputCopy, siRadius);
				biOutImage.SetPixel(siYPos, siXPos, rgbTripple);
			}
		}
	}
}

void FillImage(CBitmapImage &biInputImage, const CPixelOffset &poPixelOffset /*= CPixelOffset()*/, const CRGBTriple &trResultColor /*= CRGBTriple()*/)
{
	sizeint siImageHeight = biInputImage.GetHeight();
	sizeint siImageWidth = biInputImage.GetWidth();

	ASSERTE(poPixelOffset.siXpos < siImageWidth && poPixelOffset.siYpos < siImageHeight);

	CRGBTriple rgbFromTriple = biInputImage.GetPixel(poPixelOffset.siYpos, poPixelOffset.siXpos);

}


}; // namespace BitmapUtils
