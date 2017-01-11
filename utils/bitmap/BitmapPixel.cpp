#include "StdAfx.h"
#include "BitmapPixel.h"

//////////////////////////////////////////////////
// CRGBTriple implementation

CRGBTriple::CRGBTriple():
	m_ucBlue(0),
	m_ucGreen(0),
	m_ucRed(0) 
{
	static_assert(sizeof(CRGBTriple) == 3, "Incorrect CRGBQuad structure size"); 
}

CRGBTriple::CRGBTriple(uchar ucRed, uchar ucGreen, uchar ucBlue): 
	m_ucBlue(ucBlue),
	m_ucGreen(ucGreen),
	m_ucRed(ucRed) 
{ 
	static_assert(sizeof(CRGBTriple) == 3, "Incorrect CRGBQuad structure size"); 
}

CRGBTriple::CRGBTriple(uint uiColors)
{
	m_ucRed = (uiColors >> 16) & 0xFF;
	m_ucGreen = (uiColors >> 8) & 0xFF;
	m_ucBlue = uiColors & 0xFF;
}


void CRGBTriple::ShiftRGB(int iRed, int iGreen, int iBlue)
{
	m_ucRed = iRed > 0 ? _MIN(255, m_ucRed + iRed) : _MAX(0, m_ucRed + iRed);
	m_ucGreen = iGreen > 0 ? _MIN(255, m_ucGreen + iGreen) : _MAX(0, m_ucGreen + iGreen);
	m_ucBlue = iBlue > 0 ? _MIN(255, m_ucBlue + iBlue) : _MAX(0, m_ucBlue + iBlue);
}