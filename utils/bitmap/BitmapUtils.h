#ifndef __BITMAPUTILS_H_INCLUDED
#define __BITMAPUTILS_H_INCLUDED

#include "BitmapPixel.h"

#define MAX_WIDTH	10000

class CBitmapImage;

namespace BitmapUtils
{
	typedef CRGBTriple (*PixelBlender)(sizeint siYPos, sizeint siXPos, const CBitmapImage &bmpImage, sizeint siRadius);

	struct CPixelOffset
	{
		CPixelOffset(sizeint siY = 0, sizeint siX = 0) : siXpos(siX), siYpos(siY) { }

		bool operator<(const CPixelOffset &trOther) { return (siYpos * MAX_WIDTH + siYpos) < (trOther.siYpos * MAX_WIDTH + trOther.siYpos); }

		sizeint siXpos; 
		sizeint siYpos;
	};

	CRGBTriple AvgBlender(sizeint siYPos, sizeint siXPos, const CBitmapImage &bmpImage, sizeint siRadius);
	CRGBTriple MedianBlender(sizeint siYPos, sizeint siXPos, const CBitmapImage &bmpImage, sizeint siRadius);

	void BlendImagePixels(const CBitmapImage &biInputImage, CBitmapImage &biOutImage, PixelBlender fPixelBlender, sizeint siRadius = 1, sizeint siTimes = 1);

	void FillImage(CBitmapImage &biInputImage, const CPixelOffset &poPixelOffset = CPixelOffset(), const CRGBTriple &trResultColor = CRGBTriple());

}; // namespace BitmapUtils

#endif // __BITMAPUTILS_H_INCLUDED
