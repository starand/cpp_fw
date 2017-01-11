#ifndef __SERIALPORT_H_INCLUDED
#define __SERIALPORT_H_INCLUDED

#define ARDUINO_WAIT_TIME 2000

#include "types.h"
#include <windows.h>

class CSerialPort
{
public:
    CSerialPort();
    ~CSerialPort();

public:
	bool Connect(const string& sPortName);
	void Disconnect();

	const char* GetPortName() const { return m_sPort.c_str(); }
	bool IsConnected() const { return m_bConnected; }
	HANDLE GetHandle() const { return m_hSerial; }

	int ReadData( char *szBuffer, uint nbChar );
    bool WriteData( const char *szBuffer, uint nbChar );

private:
    HANDLE		m_hSerial;
	string		m_sPort;

    bool		m_bConnected;

    COMSTAT		m_csStatus;
	DWORD		m_dwErrors;
};

#endif // __SERIALPORT_H_INCLUDED
