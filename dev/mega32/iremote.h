//Beta
//*****

#ifndef _REMOTE_H
 #define _REMOTE_H
 /*____________________________________________________________________________________________*/

//Constants
//*********

//States
#define IR_VALIDATE_LEAD_HIGH 0
#define IR_VALIDATE_LEAD_LOW 1
#define IR_RECEIVE_BITS 3
#define IR_WAIT_STOP_BIT 4

//Others
#define TOL 0.1			//Tollerence for timming
#define QMAX 8			//Size of the Remote command buffer
#define RC_NONE 255		//This val is returned by GetRemoteCmd when no key is pressed


typedef enum
{
	EIRC_POWER      = 0x00,
	EIRC_VOLPLUS    = 0x01,
	EIRC_FUNC       = 0x02,
	EIRC_BACKWARD   = 0x04,
	EIRC_PAUSE      = 0x05,
	EIRC_FORWARD    = 0x06,
	EIRC_DOWN       = 0x08,
	EIRC_VOLMINUS   = 0x09,
	EIRC_UP         = 0x0A,
	EIRC_0          = 0x0C,
	EIRC_EQ         = 0x0D,
	EIRC_REPEAT     = 0x0E,
	EIRC_1          = 0x10,
	EIRC_2          = 0x11,
	EIRC_3          = 0x12,
	EIRC_4          = 0x14,
	EIRC_5          = 0x15,
	EIRC_6          = 0x16,
	EIRC_7          = 0x18,
	EIRC_8          = 0x19,
	EIRC_9          = 0x1A,
	
	EIRC_NOCMD		= 0x85,
	EIRC_NONE		= 0xFF
} EIRCOMMAND;

//Functions
//*********

void ResetIR();
void RemoteInit();
EIRCOMMAND GetRemoteCmd(char wait);

 /*____________________________________________________________________________________________*/
#endif