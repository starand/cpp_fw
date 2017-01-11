#include "StdAfx.h"

#include "SocketFileUtils.h"

#include "socket.h"
#include "fileutils.h"

#ifndef FILE_BUFFER_SIZE
#	define FILE_BUFFER_SIZE	0x1000
#endif

static const char g_szReadFileMode[] = "rb";
static const char g_szWriteFileMode[] = "wb";


namespace SocketFileUtils
{
	bool SendFileContent(CSocket *psSocket, const string &sFileName)
	{
		START_FUNCTION_BOOL();
		ASSERTE(psSocket && psSocket->IsConnected());

		FileUtils::CFileWrapper fwFile(sFileName.c_str(), g_szReadFileMode);
		if (!fwFile)
		{
			break;
		}

		char szBuffer[FILE_BUFFER_SIZE + 1];

		bool bAnyFault = false;

		while (!feof(fwFile))
		{
			sizeint siPartSize = fread(szBuffer, 1, FILE_BUFFER_SIZE, fwFile);
			if (!siPartSize)
			{
				bAnyFault = true;
				break;
			}

			if (!psSocket->Send(szBuffer, siPartSize))
			{
				bAnyFault = true;
				break;
			}
		}

		CHECK_ANY_FAULT();
		END_FUNCTION_BOOL();
	}

	bool RecvFileContent(CSocket *psSocket, const string &sFileName, const uint64 &uiFileSize)
	{
		START_FUNCTION_BOOL();
		ASSERTE(psSocket && psSocket->IsConnected());

		FileUtils::CFileWrapper fwFile(sFileName.c_str(), g_szWriteFileMode);
		if (!fwFile)
		{
			break;
		}

		char szBuffer[FILE_BUFFER_SIZE + 1];
		uint64 uiLeftSize = uiFileSize;

		bool bAnyFault = false;

		do 
		{
			sizeint siPartSize = (sizeint)_MIN(FILE_BUFFER_SIZE, uiLeftSize);
			if (!psSocket->Recv(szBuffer, siPartSize))
			{
				bAnyFault = true;
				break;
			}

			sizeint siBytesWritten = fwrite(szBuffer, 1, siPartSize, fwFile);
			if (siPartSize != siBytesWritten)
			{
				bAnyFault = true;
				break;
			}

			uiLeftSize -= siPartSize;
		}
		while (uiLeftSize);

		CHECK_ANY_FAULT();
		END_FUNCTION_BOOL();
	}

};
