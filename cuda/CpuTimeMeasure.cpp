#include "StdAfx.h"

#include "CpuTimeMeasure.h"
#include <helper_timer.h>

//////////////////////////////////////////////////
// CCpuTimeMeasure implementation

CCpuTimeMeasure::CCpuTimeMeasure():
	m_pwiTimer(NULL)
{
	sdkCreateTimer(&m_pwiTimer);
	sdkResetTimer(&m_pwiTimer);
	sdkStartTimer(&m_pwiTimer);
}

CCpuTimeMeasure::~CCpuTimeMeasure()
{
	sdkDeleteTimer(&m_pwiTimer);
}


float CCpuTimeMeasure::GetTimeout()
{
	sdkStopTimer(&m_pwiTimer);

	float fResult = sdkGetTimerValue(&m_pwiTimer);
	return fResult;
}
