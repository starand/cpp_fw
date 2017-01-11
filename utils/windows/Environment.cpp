#include "StdAfx.h"
#include "Environment.h"


//////////////////////////////////////////////////
// CEnvironment implementation

CEnvironment::CEnvironment()
{

}

CEnvironment::~CEnvironment()
{

}


/*static */
bool CEnvironment::GetVariable(const string &sVariableName, string &sOutValue)
{
	static const sizeint siBufferSIze = 1024;		// BE CAREFUL HERE -- An environment variable has a maximum size limit of 32,767 characters, including the null-terminating character.
	char szVariableBuffer[siBufferSIze] = { 0 };	// http://msdn.microsoft.com/en-us/library/windows/desktop/ms683188(v=vs.85).aspx

	bool bResult = false;
	if (GetEnvironmentVariableA(sVariableName.c_str(), szVariableBuffer, siBufferSIze))
	{
		sOutValue.assign(szVariableBuffer);
		bResult = true;
	}
	
	return bResult;
}
