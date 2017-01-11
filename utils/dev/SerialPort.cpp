#include "stdafx.h"
#include "SerialPort.h"

//////////////////////////////////////////////////
// CSerial implementation

CSerialPort::CSerialPort():
	m_hSerial(NULL),
	m_sPort(),
	m_bConnected(false),
	m_csStatus(),
	m_dwErrors(0)
{

}

CSerialPort::~CSerialPort()
{
	Disconnect();
}


bool CSerialPort::Connect(const string& sPortName) 
{
	START_FUNCTION_BOOL();

	m_hSerial = CreateFileA( sPortName.c_str(), GENERIC_READ | GENERIC_WRITE, 
		0, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);

    if (INVALID_HANDLE_VALUE == m_hSerial)
    {
        if (ERROR_FILE_NOT_FOUND == GetLastError())
		{
			LOG_ERROR2_BREAK("Handle was not attached. %s is not available.", sPortName.c_str());
	    }
        else
		{
			LOG_ERROR2_BREAK("Unable to connect to %s", sPortName.c_str());
		}
    }

    DCB dcbSerialParams = { 0 };

    if (!GetCommState(m_hSerial, &dcbSerialParams))
    {
        LOG_ERROR_BREAK("Unable to get current serial parameters");
    }

    // Define serial connection parameters for the arduino board
    dcbSerialParams.BaudRate = CBR_9600;
    dcbSerialParams.ByteSize = 8;
    dcbSerialParams.StopBits = ONESTOPBIT;
    dcbSerialParams.Parity = NOPARITY;

    if (!SetCommState(m_hSerial, &dcbSerialParams))
	{
		LOG_ERROR2_BREAK("Unable to set Serial Port parameters - %u", GetLastError());
    }

    m_bConnected = true;
    Sleep(ARDUINO_WAIT_TIME); // wait 2s as the arduino board will be reseting

	END_FUNCTION_BOOL();
}

void CSerialPort::Disconnect()
{
	if (m_bConnected)
	{
		CloseHandle( m_hSerial );
		m_bConnected = false;
	}
}

int CSerialPort::ReadData( char* szBuffer, unsigned int nbChar )
{
	int nResult = -1;
    DWORD nBytesRead = 0;
    uint uiToRead = 0;

    ClearCommError( m_hSerial, &m_dwErrors, &m_csStatus );

    if( m_csStatus.cbInQue > 0 )
    {
        if( m_csStatus.cbInQue > nbChar ) uiToRead = nbChar;
        uiToRead = m_csStatus.cbInQue;

        if( ReadFile(m_hSerial, szBuffer, uiToRead, &nBytesRead, NULL) && nBytesRead != 0 ) {
            nResult = nBytesRead;
			szBuffer[ nBytesRead ] = 0;
        }

    }

    return nResult;
}

bool CSerialPort::WriteData( const char* szBuffer, unsigned int nbChar )
{
	bool bResult = true;
    DWORD dwBytesSend = 0;

    if( !WriteFile( m_hSerial, (void *)szBuffer, nbChar, &dwBytesSend, 0) )
    {
        ClearCommError( m_hSerial, &m_dwErrors, &m_csStatus );
        bResult = false;
    }

    return bResult;
}
