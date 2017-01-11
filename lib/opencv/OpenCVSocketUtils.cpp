#include "StdAfx.h"

#include "OpenCVSocketUtils.h"

#include "socket.h"
#include <opencv\cv.h>


#define BLOCK_HEIGHT	8
#define BLOCK_WIDTH		8
#define BLOCK_STRIDE	24

#define IMAGE_HEIGHT(pipImage) pipImage->height
#define IMAGE_WIDTH(pipImage) pipImage->width
#define IMAGE_DATA(pipImage) pipImage->imageData

static const sizeint g_siImageHeaderSize = sizeof(IplImage);


namespace OpenCVSocketUtils
{
	bool RetrieveImageHeader(CSocket *psSocket, IplImage *pipOutImage)
	{
		ASSERTE(psSocket); ASSERTE(psSocket->IsConnected()); ASSERTE(pipOutImage);

		bool bResult = psSocket->Recv((char*)pipOutImage, g_siImageHeaderSize);
		return bResult;
	}

	bool RetrieveImageData(CSocket *psSocket, IplImage *pipVarImage)
	{
		ASSERTE(psSocket); ASSERTE(psSocket->IsConnected()); ASSERTE(pipVarImage); ASSERTE(pipVarImage->imageData);

		bool bResult = psSocket->Recv(pipVarImage->imageData, pipVarImage->imageSize);
		return bResult;
	}


	void SetImageBuffer(IplImage *pipVarImage, char *szBuffer)
	{
		ASSERTE(pipVarImage);

		pipVarImage->imageData = pipVarImage->imageDataOrigin = szBuffer;
	}


	bool SendImageHeader(CSocket *psSocket, IplImage *pipImage)
	{
		ASSERTE(psSocket); ASSERTE(psSocket->IsConnected()); ASSERTE(pipImage);

		bool bResult = psSocket->Send((char*)pipImage, g_siImageHeaderSize);
		return bResult;
	}

	bool SendImageData(CSocket *psSocket, IplImage *pipImage)
	{
		ASSERTE(psSocket); ASSERTE(psSocket->IsConnected()); ASSERTE(pipImage); ASSERTE(pipImage->imageData);

		bool bResult = psSocket->Send(pipImage->imageData, pipImage->imageSize);
		return bResult;
	}


	IplImage *CloneImage(IplImage *piInputImage)
	{
		ASSERTE(piInputImage);

		IplImage *pPrevImage = cvCloneImage(piInputImage);
		cvCopyImage(piInputImage, pPrevImage);

		return pPrevImage;
	}

	struct CUpdateBlock
	{
		CUpdateBlock(sizeint siBlockIndex) : m_siBlockIdx(siBlockIndex) { }

		sizeint	m_siBlockIdx;
		char	m_pvBlockData[BLOCK_HEIGHT * BLOCK_WIDTH];
	};

	bool SendImageDiffs(CSocket *psSocket, IplImage *pipImage, IplImage *pipPrevImage)
	{
		ASSERTE(psSocket); ASSERTE(psSocket->IsConnected()); ASSERTE(pipImage);  ASSERTE(pipPrevImage);

		sizeint siBlockIndex = 0;
		sizeint siStrideSize = IMAGE_WIDTH(pipImage) * 3;

		bool bAnyFault = false;
		for (int Y = 0; Y < IMAGE_HEIGHT(pipImage); Y += BLOCK_HEIGHT)
		{
			for (int X = 0; X < IMAGE_WIDTH(pipImage); X += BLOCK_WIDTH)
			{
				CUpdateBlock ubUpdateBlock(siBlockIndex++);

				char *szDstAddr = ubUpdateBlock.m_pvBlockData;
				char *szSrcAddr = IMAGE_DATA(pipImage) + siStrideSize * Y + X * 3;

				for (int idx = 0; idx < BLOCK_HEIGHT; ++idx, szDstAddr += BLOCK_STRIDE, szSrcAddr += siStrideSize)
				{
					memcpy(szDstAddr, szSrcAddr, BLOCK_STRIDE);
				}

				if (!psSocket->Send((char *)&ubUpdateBlock, sizeof(ubUpdateBlock)))
				{
					bAnyFault = true;
					break;
				}
			}

			if (bAnyFault)
			{
				break;
			}
		}

		bool bResult = !bAnyFault;
		return bResult;
	}

	bool RetrieveImageDiffs(CSocket *psSocket, IplImage *pipImage)
	{
		ASSERTE(psSocket); ASSERTE(psSocket->IsConnected()); ASSERTE(pipImage);

		return RetrieveImageData(psSocket, pipImage);
	}

};

