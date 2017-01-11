#ifndef __MD5_H_INCLUDED
#define __MD5_H_INCLUDED


namespace MD5
{
	string GetMD5(const string &sValue);

////////////////////////////////////////
// Implementation part
////////////////////////////////////////

	typedef struct 
	{
		uint	uiCount[2];		/* message length in bits, lsw first */
		uint	uiABCD[4];		/* digest buffer */
		uchar	szBuffer[64];	/* accumulate block */
	} CMD5State;

	/* Initialize the algorithm. */
	void MD5Init(CMD5State *pmdMD5State);

	/* Append a string to the message. */
	void MD5Append(CMD5State *pmdMD5State, const uchar *puData, int nBytes);

	/* */
	void MD5Process(CMD5State *pmdMD5State, const uchar *puData /*[64]*/);

	/* Finish the message and return the digest. */
	void MD5Finish(CMD5State *pmdMD5State, uchar aucDigest[16]);
};

#endif // __MD5_H_INCLUDED
