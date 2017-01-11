#include "StdAfx.h"
#include "CudaMD5.h"


#define HASH_SIZE	sizeof(char) * 16
#define HASH_STRING_SIZE	33


__constant__ uint g_auiTArray[64] = 
{
	0xd76aa478, 0xe8c7b756, 0x242070db, 0xc1bdceee, 0xf57c0faf, 0x4787c62a, 0xa8304613, 0xfd469501,
	0x698098d8, 0x8b44f7af, 0xffff5bb1, 0x895cd7be, 0x6b901122, 0xfd987193, 0xa679438e, 0x49b40821,
	0xf61e2562, 0xc040b340, 0x265e5a51, 0xe9b6c7aa, 0xd62f105d, 0x02441453, 0xd8a1e681, 0xe7d3fbc8,
	0x21e1cde6, 0xc33707d6, 0xf4d50d87, 0x455a14ed, 0xa9e3e905, 0xfcefa3f8, 0x676f02d9, 0x8d2a4c8a,
	0xfffa3942, 0x8771f681, 0x6d9d6122, 0xfde5380c, 0xa4beea44, 0x4bdecfa9, 0xf6bb4b60, 0xbebfbc70,
	0x289b7ec6, 0xeaa127fa, 0xd4ef3085, 0x04881d05, 0xd9d4d039, 0xe6db99e5, 0x1fa27cf8, 0xc4ac5665,
	0xf4292244, 0x432aff97, 0xab9423a7, 0xfc93a039, 0x655b59c3, 0x8f0ccc92, 0xffeff47d, 0x85845dd1,
	0x6fa87e4f, 0xfe2ce6e0, 0xa3014314, 0x4e0811a1, 0xf7537e82, 0xbd3af235, 0x2ad7d2bb, 0xeb86d391,
};

__constant__ uchar g_aucPad[64] = 
{
	0x80, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
};


typedef struct 
{
	uint	uiCount[2];		/* message length in bits, lsw first */
	uint	uiABCD[4];		/* digest buffer */
	uchar	szBuffer[64];	/* accumulate block */
} CMD5State;



void CreateDeviceInputData(const string_v &vsInputStrings, char *&d_pData)
{
	sizeint siDataSize = vsInputStrings.size();
	sizeint siTotalElemsCount = 0;

	for (auto &sValue : vsInputStrings)
	{
		siTotalElemsCount += sValue.length();
	}
	
	sizeint siHeaderSize = siDataSize * 2 * sizeof(int);
	sizeint siTotalDataSize = siHeaderSize + (siTotalElemsCount + siDataSize) * sizeof(char *); //  + siDataSize -- needed for '\0' last char

	char *pcTempData = (char *)malloc(siTotalDataSize);
	ASSERTE(pcTempData);

	int *piHeader = (int *)pcTempData;
	sizeint siDataOffset = siHeaderSize;

	for (sizeint siIndex = 0; siIndex < siDataSize; ++siIndex)
	{
		sizeint siElementLength = vsInputStrings[siIndex].length();

		piHeader[siIndex * 2] = siDataOffset;
		piHeader[siIndex * 2 + 1] = siElementLength;

		strcpy(pcTempData + siDataOffset, vsInputStrings[siIndex].c_str());

		siDataOffset += siElementLength + 1;
	}

	cudaErrorCheck(cudaMalloc(&d_pData, siTotalDataSize));
	cudaErrorCheck(cudaMemcpy(d_pData, pcTempData, siTotalDataSize, cudaMemcpyHostToDevice));

	free(pcTempData);
}

void CreateDeviceResultData(sizeint siStringsCount, char *&d_pData)
{
	sizeint siDataSize = siStringsCount * HASH_SIZE;

	cudaErrorCheck(cudaMalloc(&d_pData, siDataSize));
}

void FreeDeviceData(char *d_pData)
{
	cudaErrorCheck(cudaFree(d_pData));
}

void ConvertDeviceResultToVector(char *d_pResults, sizeint siStringsCount, string_v &vsOutResults)
{
	sizeint siHashSize = HASH_SIZE;
	sizeint siDataSize = siStringsCount * siHashSize;

	char *h_pData = (char *)malloc(siDataSize);
	ASSERTE(h_pData);

	cudaErrorCheck(cudaMemcpy(h_pData, d_pResults, siDataSize, cudaMemcpyDeviceToHost));
	
	string_v vsHashes;
	char szBuffer[33];

	uchar *pchMd5Hash = (uchar *)h_pData;
	for (sizeint idx = 0; idx < siStringsCount; ++idx)
	{
		sprintf(szBuffer, "%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x", 
			pchMd5Hash[0], pchMd5Hash[1], pchMd5Hash[2], pchMd5Hash[3], pchMd5Hash[4], pchMd5Hash[5], pchMd5Hash[6], pchMd5Hash[7], 
			pchMd5Hash[8], pchMd5Hash[9], pchMd5Hash[10], pchMd5Hash[11], pchMd5Hash[12], pchMd5Hash[13], pchMd5Hash[14], pchMd5Hash[15]);

		vsHashes.push_back(szBuffer);
		pchMd5Hash += siHashSize;
	}

	free(h_pData);
	vsOutResults.swap(vsHashes);
}


__device__ void MD5Init(CMD5State *pmdMD5State)
{
	pmdMD5State->uiCount[0] = pmdMD5State->uiCount[1] = 0;

	pmdMD5State->uiABCD[0] = 0x67452301;
	pmdMD5State->uiABCD[1] = 0xefcdab89;
	pmdMD5State->uiABCD[2] = 0x98badcfe;
	pmdMD5State->uiABCD[3] = 0x10325476;
}

__device__ void MD5Process(CMD5State *pmdMD5State, const uchar *pucData /*[64]*/)
{
	uint a = pmdMD5State->uiABCD[0], b = pmdMD5State->uiABCD[1];
	uint c = pmdMD5State->uiABCD[2], d = pmdMD5State->uiABCD[3];
	uint t;

#ifdef ARCH_IS_BIG_ENDIAN
	/* On big-endian machines, we must arrange the bytes in the right order.  (This also works on machines of unknown byte order.) */
	uint puiX[16];
	const uchar *pucXP = pucData;

	for (sizeint idx = 0; idx < 16; ++idx, pucXP += 4)
	{
		puiX[idx] = pucXP[0] + (pucXP[1] << 8) + (pucXP[2] << 16) + (pucXP[3] << 24);
	}
#else
	/* On little-endian machines, we can process properly aligned data without copying it. */
	uint auiXBuf[16];
	const uint *puiX;

	if (!((pucData - (const uchar *)0) & 3))
	{
		puiX = (const uint *)pucData; /* data are properly aligned */
	}
	else
	{
		memcpy(auiXBuf, pucData, 64); /* not aligned */
		puiX = auiXBuf;
	}
#endif

#define ROTATE_LEFT(x, n) (((x) << (n)) | ((x) >> (32 - (n))))

/* Round 1. Let [abcd k s i] denote the operation -- a = b + ((a + F(b,c,d) + X[k] + T[i]) <<< s). */
#define F(x, y, z) (((x) & (y)) | (~(x) & (z)))

#define SET(a, b, c, d, k, s, Ti) \
	t = a + F(b,c,d) + puiX[k] + Ti; \
	a = ROTATE_LEFT(t, s) + b

	/* Do the following 16 operations. */
	SET(a, b, c, d,  0,  7,  g_auiTArray[0]);
	SET(d, a, b, c,  1, 12,  g_auiTArray[1]);
	SET(c, d, a, b,  2, 17,  g_auiTArray[2]);
	SET(b, c, d, a,  3, 22,  g_auiTArray[3]);
	SET(a, b, c, d,  4,  7,  g_auiTArray[4]);
	SET(d, a, b, c,  5, 12,  g_auiTArray[5]);
	SET(c, d, a, b,  6, 17,  g_auiTArray[6]);
	SET(b, c, d, a,  7, 22,  g_auiTArray[7]);
	SET(a, b, c, d,  8,  7,  g_auiTArray[8]);
	SET(d, a, b, c,  9, 12, g_auiTArray[9]);
	SET(c, d, a, b, 10, 17, g_auiTArray[10]);
	SET(b, c, d, a, 11, 22, g_auiTArray[11]);
	SET(a, b, c, d, 12,  7, g_auiTArray[12]);
	SET(d, a, b, c, 13, 12, g_auiTArray[13]);
	SET(c, d, a, b, 14, 17, g_auiTArray[14]);
	SET(b, c, d, a, 15, 22, g_auiTArray[15]);
#undef SET

/* Round 2.  Let [abcd k s i] denote the operation --  a = b + ((a + G(b,c,d) + X[k] + T[i]) <<< s). */
#define G(x, y, z) (((x) & (z)) | ((y) & ~(z)))
#define SET(a, b, c, d, k, s, Ti)\
	t = a + G(b,c,d) + puiX[k] + Ti;\
	a = ROTATE_LEFT(t, s) + b
		/* Do the following 16 operations. */
	SET(a, b, c, d,  1,  5, g_auiTArray[16]);
	SET(d, a, b, c,  6,  9, g_auiTArray[17]);
	SET(c, d, a, b, 11, 14, g_auiTArray[18]);
	SET(b, c, d, a,  0, 20, g_auiTArray[19]);
	SET(a, b, c, d,  5,  5, g_auiTArray[20]);
	SET(d, a, b, c, 10,  9, g_auiTArray[21]);
	SET(c, d, a, b, 15, 14, g_auiTArray[22]);
	SET(b, c, d, a,  4, 20, g_auiTArray[23]);
	SET(a, b, c, d,  9,  5, g_auiTArray[24]);
	SET(d, a, b, c, 14,  9, g_auiTArray[25]);
	SET(c, d, a, b,  3, 14, g_auiTArray[26]);
	SET(b, c, d, a,  8, 20, g_auiTArray[27]);
	SET(a, b, c, d, 13,  5, g_auiTArray[28]);
	SET(d, a, b, c,  2,  9, g_auiTArray[29]);
	SET(c, d, a, b,  7, 14, g_auiTArray[30]);
	SET(b, c, d, a, 12, 20, g_auiTArray[31]);
#undef SET

/* Round 3.  Let [abcd k s t] denote the operation --  a = b + ((a + H(b,c,d) + X[k] + T[i]) <<< s). */
#define H(x, y, z) ((x) ^ (y) ^ (z))
#define SET(a, b, c, d, k, s, Ti)\
	t = a + H(b,c,d) + puiX[k] + Ti;\
	a = ROTATE_LEFT(t, s) + b
		/* Do the following 16 operations. */
	SET(a, b, c, d,  5,  4, g_auiTArray[32]);
	SET(d, a, b, c,  8, 11, g_auiTArray[33]);
	SET(c, d, a, b, 11, 16, g_auiTArray[34]);
	SET(b, c, d, a, 14, 23, g_auiTArray[35]);
	SET(a, b, c, d,  1,  4, g_auiTArray[36]);
	SET(d, a, b, c,  4, 11, g_auiTArray[37]);
	SET(c, d, a, b,  7, 16, g_auiTArray[38]);
	SET(b, c, d, a, 10, 23, g_auiTArray[39]);
	SET(a, b, c, d, 13,  4, g_auiTArray[40]);
	SET(d, a, b, c,  0, 11, g_auiTArray[41]);
	SET(c, d, a, b,  3, 16, g_auiTArray[42]);
	SET(b, c, d, a,  6, 23, g_auiTArray[43]);
	SET(a, b, c, d,  9,  4, g_auiTArray[44]);
	SET(d, a, b, c, 12, 11, g_auiTArray[45]);
	SET(c, d, a, b, 15, 16, g_auiTArray[46]);
	SET(b, c, d, a,  2, 23, g_auiTArray[47]);
#undef SET

/* Round 4. Let [abcd k s t] denote the operation -- a = b + ((a + I(b,c,d) + X[k] + T[i]) <<< s). */
#define I(x, y, z) ((y) ^ ((x) | ~(z)))
#define SET(a, b, c, d, k, s, Ti)\
	t = a + I(b,c,d) + puiX[k] + Ti;\
	a = ROTATE_LEFT(t, s) + b
		/* Do the following 16 operations. */
	SET(a, b, c, d,  0,  6, g_auiTArray[48]);
	SET(d, a, b, c,  7, 10, g_auiTArray[49]);
	SET(c, d, a, b, 14, 15, g_auiTArray[50]);
	SET(b, c, d, a,  5, 21, g_auiTArray[51]);
	SET(a, b, c, d, 12,  6, g_auiTArray[52]);
	SET(d, a, b, c,  3, 10, g_auiTArray[53]);
	SET(c, d, a, b, 10, 15, g_auiTArray[54]);
	SET(b, c, d, a,  1, 21, g_auiTArray[55]);
	SET(a, b, c, d,  8,  6, g_auiTArray[56]);
	SET(d, a, b, c, 15, 10, g_auiTArray[57]);
	SET(c, d, a, b,  6, 15, g_auiTArray[58]);
	SET(b, c, d, a, 13, 21, g_auiTArray[59]);
	SET(a, b, c, d,  4,  6, g_auiTArray[60]);
	SET(d, a, b, c, 11, 10, g_auiTArray[61]);
	SET(c, d, a, b,  2, 15, g_auiTArray[62]);
	SET(b, c, d, a,  9, 21, g_auiTArray[63]);
#undef SET

		/* Then perform the following additions. (That is increment each
		of the four registers by the value it had before this block was started.) */
	pmdMD5State->uiABCD[0] += a;
	pmdMD5State->uiABCD[1] += b;
	pmdMD5State->uiABCD[2] += c;
	pmdMD5State->uiABCD[3] += d;
}

__device__ void MD5Append(CMD5State *pmdMD5State, const uchar *aucData, int nBytes)
{
	do 
	{
		const uchar *pData = aucData;

		int nLeftBytes = nBytes;
		int nOffset = (pmdMD5State->uiCount[0] >> 3) & 63;
		uint uiBits = (uint)(nBytes << 3);

		if (nBytes <= 0)
		{
			break;
		}

		/* Update the message length. */
		pmdMD5State->uiCount[1] += nBytes >> 29;
		pmdMD5State->uiCount[0] += uiBits;

		if (pmdMD5State->uiCount[0] < uiBits)
		{
			pmdMD5State->uiCount[1]++;
		}

		/* Process an initial partial block. */
		if (nOffset)
		{
			int nCopy = (nOffset + nBytes > 64 ? 64 - nOffset : nBytes);
			memcpy(pmdMD5State->szBuffer + nOffset, pData, nCopy);

			if (nOffset + nCopy < 64)
			{
				break;
			}

			pData += nCopy;
			nLeftBytes -= nCopy;

			MD5Process(pmdMD5State, pmdMD5State->szBuffer);
		}

		/* Process full blocks. */
		for (; nLeftBytes >= 64; pData += 64, nLeftBytes -= 64)
		{
			MD5Process(pmdMD5State, pData);
		}

		/* Process a final partial block. */
		if (nLeftBytes)
		{
			memcpy(pmdMD5State->szBuffer, pData, nLeftBytes);
		}
	}
	while (false);
}

__device__ void MD5Finish(CMD5State *pmdMD5State, uchar aucDigest[16])
{
	uchar aucData[8];

	/* Save the length before padding. */
	for (sizeint i = 0; i < 8; ++i)
	{
		aucData[i] = (uchar)(pmdMD5State->uiCount[i >> 2] >> ((i & 3) << 3));
	}
		
	/* Pad to 56 bytes mod 64. */
	MD5Append(pmdMD5State, g_aucPad, ((55 - (pmdMD5State->uiCount[0] >> 3)) & 63) + 1);
	/* Append the length. */
	MD5Append(pmdMD5State, aucData, 8);

	for (sizeint i = 0; i < 16; ++i)
	{
		aucDigest[i] = (uchar)(pmdMD5State->uiABCD[i >> 2] >> ((i & 3) << 3));
	}
}

__device__ void CalculateMD5(const char *szData, char *pcResultData, uint uiLength)
{
	CMD5State msMD5State;

	MD5Init(&msMD5State);
	MD5Append(&msMD5State, (const uchar *)szData, uiLength);

	const int siHashSize = HASH_SIZE;
	uchar aucDigest[siHashSize];
	MD5Finish(&msMD5State, aucDigest);

	for (int idx = 0; idx < siHashSize; ++idx)
	{
		pcResultData[idx] = aucDigest[idx];
	}

	//printf("%2.2x%2.2x%2.2x%2.2x%2.2x%2.2x%2.2x%2.2x%2.2x%2.2x%2.2x%2.2x%2.2x%2.2x%2.2x%2.2x\n",
	//	aucDigest[0], aucDigest[1], aucDigest[2], aucDigest[3], aucDigest[4], aucDigest[5], aucDigest[6], aucDigest[7], 
	//	aucDigest[8], aucDigest[9], aucDigest[10], aucDigest[11], aucDigest[12], aucDigest[13], aucDigest[14], aucDigest[15]);
}

__global__ void CalculateMd5Kernel(const char *pcData, char *pcResultData, sizeint siDataSize)
{
	int tid = threadIdx.x;

	if (tid < siDataSize)
	{
		int idx = tid << 1;

		uint uiOffset = ((int *)pcData)[idx];
		uint uiLength = ((int *)pcData)[idx + 1];

		const char *pszCurrentString = pcData + uiOffset;
		char *pszResultData = pcResultData + tid * HASH_SIZE;

		CalculateMD5(pszCurrentString, pszResultData, uiLength);
	}
}


void CalculateMd5(const string_v &vsInputStrings, string_v &vsResults)
{
	char *d_pData = NULL, *d_pResults = NULL;

	sizeint siStringsCount = vsInputStrings.size();

	CreateDeviceInputData(vsInputStrings, d_pData);
	CreateDeviceResultData(siStringsCount, d_pResults);
	ASSERTE(d_pData && d_pResults);

	CalculateMd5Kernel<<<1, siStringsCount>>>(d_pData, d_pResults, siStringsCount);
	
	cudaErrorCheck(cudaGetLastError());
	cudaErrorCheck(cudaDeviceSynchronize());

	ConvertDeviceResultToVector(d_pResults, siStringsCount, vsResults);
	
	FreeDeviceData(d_pData);
	FreeDeviceData(d_pResults);
}
