#ifndef __CPUTIMEMEASURE_H_INCLUDED
#define __CPUTIMEMEASURE_H_INCLUDED


class StopWatchInterface;


//////////////////////////////////////////////////
// CCpuTimeMeasure declaration

class CCpuTimeMeasure
{
public:
	CCpuTimeMeasure();
	~CCpuTimeMeasure();

public:
	float GetTimeout();

private:
	StopWatchInterface	*m_pwiTimer;
};

#endif // __CPUTIMEMEASURE_H_INCLUDED
