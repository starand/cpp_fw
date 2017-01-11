#ifndef __CUDABLENDERUTILS_H_INCLUDED
#define __CUDABLENDERUTILS_H_INCLUDED


class CBitmapImage;
struct CRGBPixel;

namespace CudaImageUtils
{
	void MedianFilter(const CBitmapImage &biImage, CBitmapImage &biOuImage, size_t siRadius);
	void EdgeDetector(const CBitmapImage &biImage, CBitmapImage &biOutImage, sizeint siPixelsDifference);

	void OverlayFilter(CBitmapImage &biSourceImage, const CBitmapImage &biFilterImage);
}


#endif // __CUDABLENDERUTILS_H_INCLUDED
