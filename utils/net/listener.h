#ifndef __H_LISTENER__
#define __H_LISTENER__

#include "thread.h"
#include "socket.h"

#define DEF_BIND_ADDRESS "0.0.0.0"

//////////////////////////////////////////////////
// ITcpServer definition

class ITcpServer 
{
public:
	virtual ~ITcpServer() { }
	virtual bool Start( ushort nPort, const string& sBindAddress ) = 0;
	virtual CSocket* GetSocket() const = 0;
};


//////////////////////////////////////////////////
// CListener definition

class CListener : public CThread, public ITcpServer
{
public:
	CListener( bool bDeleteSocket = false );
	~CListener();

	bool Start( ushort nPort, const string& sBindAddress = DEF_BIND_ADDRESS );
	CSocket* GetSocket() const { return m_pSocket; }

protected:
	bool Execute();
	virtual void OnClientAccepted( CSocket* psClient ) = 0;
	

private:
	CSocket*	m_pSocket;
	bool		m_bDeleteSocket;
};

#endif // __H_LISTENER__
