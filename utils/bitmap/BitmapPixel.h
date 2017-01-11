#ifndef __BITMAPPIXEL_H_INCLUDED
#define __BITMAPPIXEL_H_INCLUDED

//////////////////////////////////////////////////
// CRGBTriple declaration

class CRGBTriple
{
public:
	CRGBTriple();
	CRGBTriple(uchar ucRed, uchar ucGreen, uchar ucBlue);
	CRGBTriple(uint uiColors);

public:
	uchar &Red() const { return const_cast<uchar &>(m_ucRed); }
	uchar &Green() const { return const_cast<uchar &>(m_ucGreen); }
	uchar &Blue() const { return const_cast<uchar &>(m_ucBlue); }

public:
	bool operator==(const CRGBTriple &trAnotherTriple) const { return 0 == memcmp(this, &trAnotherTriple, sizeof(trAnotherTriple)); }

public:
	void ShiftRGB(int iRed, int iGreen, int iBlue);

private:
	uchar	m_ucBlue;
	uchar	m_ucGreen;
	uchar	m_ucRed;
};

#endif // __BITMAPPIXEL_H_INCLUDED
