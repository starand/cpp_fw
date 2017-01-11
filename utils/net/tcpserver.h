#ifndef __H_TCPSERVER__
#define __H_TCPSERVER__

#include "thread.h"
#include "listener.h"

class CTCPServer;

//////////////////////////////////////////////////
// CTCPWorker definition

class CTCPWorker : public CThread
{
public:
	CTCPWorker( CSocket* pSocket, CTCPServer* pTCPServer );
	~CTCPWorker();

	bool Execute();

private:
	const CSocket*		m_pSocket;
	const CTCPServer*	m_pTCPServer;
};

//////////////////////////////////////////////////
// CTCPServer definition

class CTCPServer : public CListener
{
public:
	virtual void OnClientRequest( CSocket* pClient ) = 0; 

protected:
	void OnClientAccepted( CSocket* psClient );
};

#endif // __H_TCPSERVER__
