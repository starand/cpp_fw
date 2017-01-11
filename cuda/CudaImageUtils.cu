#include "StdAfx.h"

#include "CudaImageUtils.h"
#include "BitmapImage.h"


#define BLOCK_DIM	32


namespace CudaImageUtils
{
	struct CRGBPixel
	{
		__device__ CRGBPixel() : ucRed(0), ucGreen(0), ucBlue(0) { }
		__device__ CRGBPixel(uchar ucR, uchar ucG, uchar ucB) : ucRed(ucR), ucGreen(ucG), ucBlue(ucB) { }

		__device__ bool GetIsBlack() const { return ucRed == 0 && ucGreen == 0 && ucBlue == 0; }

		__device__ void SetPixel(uchar ucR = 0, uchar ucG = 0, uchar ucB = 0) { ucRed =ucR; ucGreen = ucG; ucBlue = ucB; }

		uchar ucRed;
		uchar ucGreen;
		uchar ucBlue;
	};


	__device__ void SimpleSort(CRGBPixel *pcArray, sizeint siArraySize)
	{
		uchar ucTemp;
		for (sizeint i = 0; i < siArraySize - 1; ++i)
		{
			for (sizeint k = i + 1; k < siArraySize; ++k)
			{
				if (pcArray[k].ucRed < pcArray[i].ucRed)
				{
					ucTemp = pcArray[k].ucRed;
					pcArray[k].ucRed = pcArray[i].ucRed;
					pcArray[i].ucRed = ucTemp;
				}

				if (pcArray[k].ucGreen < pcArray[i].ucGreen)
				{
					ucTemp = pcArray[k].ucGreen;
					pcArray[k].ucGreen = pcArray[i].ucGreen;
					pcArray[i].ucGreen = ucTemp;
				}

				if (pcArray[k].ucBlue < pcArray[i].ucBlue)
				{
					ucTemp = pcArray[k].ucBlue;
					pcArray[k].ucBlue = pcArray[i].ucBlue;
					pcArray[i].ucBlue = ucTemp;
				}
			}
		}
	}

	__device__ CRGBPixel *GetPixel(const uchar *pvImageData, sizeint siXpos, sizeint siYpos, sizeint siStrideSize)
	{
		return (CRGBPixel *)(pvImageData + siYpos * siStrideSize + siXpos * 3);
	}

	__device__ void MedianFilterProcessPixel(uchar *pvImageData, uchar *pvOutImageData, sizeint siXpos, sizeint siYpos, sizeint siImageHeight, sizeint siImageWidth, sizeint siRadius)
	{
		CRGBPixel apLocalArray[225];

		sizeint siStrideSize = ALIGN_UP(siImageWidth * 3, 4);

		sizeint siMinX = _MAX((int)siXpos - (int)siRadius, 0);
		sizeint siMaxX = _MIN((int)siXpos + (int)siRadius, siImageWidth - 1);
		sizeint siMinY = _MAX((int)siYpos - (int)siRadius, 0);
		sizeint siMaxY = _MIN((int)siYpos + (int)siRadius, siImageHeight - 1);

		sizeint siPixelsPerLine = (siMaxX - siMinX + 1);
		sizeint siPixelsPerCol = (siMaxY - siMinY + 1);

		CRGBPixel *ppBufferPtr = apLocalArray;
		for (sizeint Y = siMinY; Y <= siMaxY; ++Y)
		{
			const CRGBPixel *ppCurrentPixel = GetPixel(pvImageData, siMinX, Y, siStrideSize);
			for (sizeint i = 0; i < siPixelsPerLine; ++i)
			{
				*ppBufferPtr++ = *ppCurrentPixel++;
			}
		}

		sizeint siArraySize = siPixelsPerLine * siPixelsPerCol;
		SimpleSort(apLocalArray, siArraySize);

		sizeint siMedianPos = siArraySize / 2;
		CRGBPixel *ppOutPixel = (CRGBPixel *)GetPixel(pvOutImageData, siXpos, siYpos, siStrideSize);

		*ppOutPixel = apLocalArray[siMedianPos];
	}

	__global__ void MedianFilterProcessImage(uchar *pvImageData, uchar *pvOutImageData, sizeint siImageHeight, sizeint siImageWidth, sizeint siRadius)
	{
		sizeint Y = threadIdx.y + blockDim.y * blockIdx.y;

		sizeint siHorizGridSize = gridDim.x * blockDim.x;
		sizeint siVertGridSize = gridDim.y * blockDim.y;

		while (Y < siImageHeight)
		{
			sizeint X = threadIdx.x + blockDim.x * blockIdx.x;

			while (X < siImageWidth)	
			{
				MedianFilterProcessPixel(pvImageData, pvOutImageData, X, Y, siImageHeight, siImageWidth, siRadius);
				X += siHorizGridSize;
			}

			Y += siVertGridSize;
		}
	}

	void MedianFilter(const CBitmapImage &biImage, CBitmapImage &biOutImage, sizeint siRadius)
	{
		biOutImage.CloneImage(biImage, false);

		sizeint siImageHeight = biImage.GetHeight();
		sizeint siImageWidth = biImage.GetWidth();

		sizeint siDataSize = biImage.GetDataSize();
		void *pvImageData = biImage.GetImageData();
		void *pvOutImageData = biOutImage.GetImageData();

		uchar *d_pvData, *d_pvOutData;

		cudaErrorCheck(cudaMalloc(&d_pvData, siDataSize));
		cudaErrorCheck(cudaMalloc(&d_pvOutData, siDataSize));

		cudaErrorCheck(cudaMemcpy(d_pvData, pvImageData, siDataSize, cudaMemcpyHostToDevice));

		dim3 d3BlockSize(BLOCK_DIM, BLOCK_DIM);
		dim3 d3GrideSize(_MAX(siImageHeight / d3BlockSize.y, 1), _MAX(siImageWidth / d3BlockSize.x, 1));

		sizeint siStrideLength = biImage.GetLineSize();

		ASSERTE(siRadius <= (sizeint)8 && siRadius >= (sizeint)1);

		MedianFilterProcessImage<<<d3GrideSize, d3BlockSize>>>(d_pvData, d_pvOutData, siImageHeight, siImageWidth, siRadius);
		cudaErrorCheck(cudaGetLastError());

		cudaErrorCheck(cudaMemcpy(pvOutImageData, d_pvOutData, siDataSize, cudaMemcpyDeviceToHost));

		cudaErrorCheck(cudaFree(d_pvData));
		cudaErrorCheck(cudaFree(d_pvOutData));
	}


	__device__ bool PixelsDiffer(CRGBPixel *ppFirstPixel, CRGBPixel *ppSecondPixel, sizeint siPixelsDifference)
	{
		sizeint siDifference = abs(ppFirstPixel->ucRed - ppSecondPixel->ucRed) + abs(ppFirstPixel->ucGreen - ppSecondPixel->ucGreen) + abs(ppFirstPixel->ucBlue - ppSecondPixel->ucBlue);

		return siDifference > siPixelsDifference;
	}

	__device__ void EdgeDetectorProcessPixel(uchar *pvImageData, uchar *pvOutImageData, sizeint siXpos, sizeint siYpos, sizeint siImageHeight, sizeint siImageWidth, sizeint siPixelsDifference)
	{
		sizeint siStrideSize = ALIGN_UP(siImageWidth * 3, 4);

		CRGBPixel *ppCenterPixel = GetPixel(pvImageData, siXpos, siYpos, siStrideSize);

		CRGBPixel *ppLeftPixel = siXpos ? ppCenterPixel - 1 : ppCenterPixel;
		CRGBPixel *ppRightPixel = siXpos < siImageWidth - 1 ? ppCenterPixel + 1 : ppCenterPixel;

		sizeint siUpPos = siYpos ? siYpos - 1 : siYpos;
		sizeint siDownPos = siYpos == siImageHeight - 1 ? siYpos : siYpos + 1;
		CRGBPixel *ppUpPixel = GetPixel(pvImageData, siXpos, siUpPos, siStrideSize);
		CRGBPixel *ppDownPixel = GetPixel(pvImageData, siXpos, siDownPos, siStrideSize);

		CRGBPixel *ppOutPixel = GetPixel(pvOutImageData, siXpos, siYpos, siStrideSize);

		if (PixelsDiffer(ppCenterPixel, ppLeftPixel, siPixelsDifference) || PixelsDiffer(ppCenterPixel, ppRightPixel, siPixelsDifference)
			|| PixelsDiffer(ppCenterPixel, ppUpPixel, siPixelsDifference) || PixelsDiffer(ppCenterPixel, ppDownPixel, siPixelsDifference))
		{
			*ppOutPixel = CRGBPixel();
		}
		else
		{
			//*ppOutPixel = *ppCenterPixel;
			*ppOutPixel = CRGBPixel(255, 255, 255);
		}
	}

	__global__ void EdgeDetectorProcessImage(uchar *pvImageData, uchar *pvOutImageData, sizeint siImageHeight, sizeint siImageWidth, sizeint siPixelsDifference)
	{
		sizeint Y = threadIdx.y + blockDim.y * blockIdx.y;

		sizeint siHorizGridSize = gridDim.x * blockDim.x;
		sizeint siVertGridSize = gridDim.y * blockDim.y;

		while (Y < siImageHeight)
		{
			sizeint X = threadIdx.x + blockDim.x * blockIdx.x;

			while (X < siImageWidth)	
			{
				EdgeDetectorProcessPixel(pvImageData, pvOutImageData, X, Y, siImageHeight, siImageWidth, siPixelsDifference);
				X += siHorizGridSize;
			}

			Y += siVertGridSize;
		}
	}

	void EdgeDetector(const CBitmapImage &biImage, CBitmapImage &biOutImage, sizeint siPixelsDifference)
	{
		biOutImage.CloneImage(biImage, false);

		sizeint siImageHeight = biImage.GetHeight();
		sizeint siImageWidth = biImage.GetWidth();

		sizeint siDataSize = biImage.GetDataSize();
		void *pvImageData = biImage.GetImageData();
		void *pvOutImageData = biOutImage.GetImageData();

		uchar *d_pvData, *d_pvOutData;

		cudaErrorCheck(cudaMalloc(&d_pvData, siDataSize));
		cudaErrorCheck(cudaMalloc(&d_pvOutData, siDataSize));

		cudaErrorCheck(cudaMemcpy(d_pvData, pvImageData, siDataSize, cudaMemcpyHostToDevice));

		dim3 d3BlockSize(BLOCK_DIM, BLOCK_DIM);
		dim3 d3GrideSize(_MAX(siImageHeight / d3BlockSize.y, 1), _MAX(siImageWidth / d3BlockSize.x, 1));

		sizeint siStrideLength = biImage.GetLineSize();
		EdgeDetectorProcessImage<<<d3GrideSize, d3BlockSize>>>(d_pvData, d_pvOutData, siImageHeight, siImageWidth, siPixelsDifference);
		cudaErrorCheck(cudaGetLastError());

		cudaErrorCheck(cudaMemcpy(pvOutImageData, d_pvOutData, siDataSize, cudaMemcpyDeviceToHost));
		cudaErrorCheck(cudaFree(d_pvData));
		cudaErrorCheck(cudaFree(d_pvOutData));
	}


	__device__ void OverlayFilterProcessPixel(uchar *pvImageData, const uchar *pvFilterImageData, sizeint siXpos, sizeint siYpos, sizeint siImageHeight, sizeint siImageWidth)
	{
		sizeint siStrideSize = ALIGN_UP(siImageWidth * 3, 4);
		
		CRGBPixel *ppSrcPixel = GetPixel(pvImageData, siXpos, siYpos, siStrideSize);
		const CRGBPixel * ppFilterPixel = GetPixel(pvFilterImageData, siXpos, siYpos, siStrideSize);

		if (ppFilterPixel->GetIsBlack())
		{
			ppSrcPixel->SetPixel(0xFF, 0xFF, 0xFF);
		}
	}

	__global__ void OverlapFilterProcessImage(uchar *pvImageData, const uchar *pvOutImageData, sizeint siImageHeight, sizeint siImageWidth)
	{
		sizeint Y = threadIdx.y + blockDim.y * blockIdx.y;

		sizeint siHorizGridSize = gridDim.x * blockDim.x;
		sizeint siVertGridSize = gridDim.y * blockDim.y;

		while (Y < siImageHeight)
		{
			sizeint X = threadIdx.x + blockDim.x * blockIdx.x;

			while (X < siImageWidth)	
			{
				OverlayFilterProcessPixel(pvImageData, pvOutImageData, X, Y, siImageHeight, siImageWidth);
				X += siHorizGridSize;
			}

			Y += siVertGridSize;
		}
	}

	// filter image should be black-white mask
	void OverlayFilter(CBitmapImage &biSourceImage, const CBitmapImage &biFilterImage)
	{
		sizeint siSrcImageHeight = biSourceImage.GetHeight();
		sizeint siSrcImageWidth = biSourceImage.GetWidth();
		sizeint siSrcImageDataSize = biSourceImage.GetDataSize();

		sizeint siFilterImageHeight = biFilterImage.GetHeight();
		sizeint siFilterImageWidth = biFilterImage.GetWidth();
		sizeint siFilterImageDataSize = biFilterImage.GetDataSize();

		ASSERTE(siSrcImageHeight == siFilterImageHeight && siSrcImageWidth == siFilterImageWidth && siSrcImageDataSize == siFilterImageDataSize);

		uchar *d_pvSrcData, *d_pvFilterData;

		cudaErrorCheck(cudaMalloc(&d_pvSrcData, siSrcImageDataSize));
		cudaErrorCheck(cudaMalloc(&d_pvFilterData, siFilterImageDataSize));

		void *pvSrcImageData = biSourceImage.GetImageData();
		void *pvFilterData = biFilterImage.GetImageData();

		cudaErrorCheck(cudaMemcpy(d_pvSrcData, pvSrcImageData, siSrcImageDataSize, cudaMemcpyHostToDevice));
		cudaErrorCheck(cudaMemcpy(d_pvFilterData, pvFilterData, siFilterImageDataSize, cudaMemcpyHostToDevice));

		dim3 d3BlockSize(BLOCK_DIM, BLOCK_DIM);
		dim3 d3GrideSize(_MAX(siSrcImageHeight / d3BlockSize.y, 1), _MAX(siSrcImageWidth / d3BlockSize.x, 1));

		sizeint siStrideLength = biSourceImage.GetLineSize();
		OverlapFilterProcessImage<<<d3GrideSize, d3BlockSize>>>(d_pvSrcData, d_pvFilterData, siSrcImageHeight, siSrcImageWidth);
		cudaErrorCheck(cudaGetLastError());

		cudaErrorCheck(cudaMemcpy(pvSrcImageData, d_pvSrcData, siSrcImageDataSize, cudaMemcpyDeviceToHost));
		cudaErrorCheck(cudaFree(d_pvSrcData));
		cudaErrorCheck(cudaFree(d_pvFilterData));

	}
};
