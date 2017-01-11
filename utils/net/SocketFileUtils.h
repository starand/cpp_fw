#ifndef __SOCKETUTILS_H_INCLUDED
#define __SOCKETUTILS_H_INCLUDED

//////////////////////////////////////////////////
// CSocketUtils declaration

class CSocket;

namespace SocketFileUtils
{
	bool SendFileContent(CSocket *psSocket, const string &sFileName);
	bool RecvFileContent(CSocket *psSocket, const string &sFileName, const uint64 &uiFileSize);
};

#endif // __SOCKETUTILS_H_INCLUDED
