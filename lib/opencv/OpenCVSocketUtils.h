#ifndef __IMAGESOCKETUTILS_H_INCLUDED
#define __IMAGESOCKETUTILS_H_INCLUDED

#include <opencv2\core\types_c.h>


class CSocket;

namespace OpenCVSocketUtils
{
	bool RetrieveImageHeader(CSocket *psSocket, IplImage *pipOutImage);
	bool RetrieveImageData(CSocket *psSocket, IplImage *pipVarImage);

	void SetImageBuffer(IplImage *pipVarImage, char *szBuffer);

	bool SendImageHeader(CSocket *psSocket, IplImage *pipImage);
	bool SendImageData(CSocket *psSocket, IplImage *pipImage);

	IplImage *CloneImage(IplImage *piInputImage);

	bool SendImageDiffs(CSocket *psSocket, IplImage *pipImage, IplImage *pipPrevImage);
	bool RetrieveImageDiffs(CSocket *psSocket, IplImage *pipImage);
};

#define RetrieveImageUpdates RetrieveImageData

#endif // __IMAGESOCKETUTILS_H_INCLUDED
