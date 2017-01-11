#include "StdAfx.h"
#include "MD5.h"


#ifdef BIG_ENDIAN_ARCH
#	define ARCH_IS_BIG_ENDIAN 1	/* slower, default implementation */
#endif


namespace MD5
{
	string GetMD5(const string &sValue)
	{
		CMD5State msMD5State;

		MD5Init(&msMD5State);
		MD5Append(&msMD5State, (const uchar *)sValue.c_str(), sValue.length());

		uchar aucDigest[16];
		MD5Finish(&msMD5State, aucDigest);

		string sResult;
		for (sizeint siIndex = 0; siIndex < 16; ++siIndex)
		{
			sResult += StrUtils::IntToHex(aucDigest[siIndex]);
		}

		return sResult;
	}


////////////////////
// Implementation //
////////////////////

	/* End of T computation program. */
	static const uint g_auiTArray[64] = 
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

	static const uchar g_aucPad[64] = 
	{
		0x80, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
	};


	void MD5Init(CMD5State *pmdMD5State)
	{
		pmdMD5State->uiCount[0] = pmdMD5State->uiCount[1] = 0;

		pmdMD5State->uiABCD[0] = 0x67452301;
		pmdMD5State->uiABCD[1] = 0xefcdab89;
		pmdMD5State->uiABCD[2] = 0x98badcfe;
		pmdMD5State->uiABCD[3] = 0x10325476;
	}

	void MD5Append(CMD5State *pmdMD5State, const uchar *aucData, int nBytes)
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

	void MD5Process(CMD5State *pmdMD5State, const uchar *pucData /*[64]*/)
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

	void MD5Finish(CMD5State *pmdMD5State, uchar aucDigest[16])
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
};
