#ifndef __H_PROCESSES__
#define __H_PROCESSES__

#include "taskinterface.h"

bool RunCommandLine( const char* szCommand );
bool StartProcess( const char* szCommand );
bool ExecuteCommand( const string& sCommandLine, bool bHide = false );

class CProcessTask : public ITask
{
public:
	CProcessTask( const char* szTaskCmd );
	bool Execute();

	void GetDescription( string& sDescription ) const;
private:
	string	m_sTaskCommand;
};

#endif // __H_PROCESSES__
