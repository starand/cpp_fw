#ifndef __H_WORKER__
#define __H_WORKER__

#include "thread.h"
#include "socket.h"

#define DECLARE_CONSTRUCTOR( NAME ) NAME( CSocket* pSocket ) : CWorker(pSocket) { }

class CWorker : public CThread
{
public:
	CWorker( CSocket* pSocket );
	~CWorker();

protected:
	 bool Execute();
	 virtual void OnExecute() = 0;

	const CSocket* m_pSocket;
};

#endif // __H_WORKER__
