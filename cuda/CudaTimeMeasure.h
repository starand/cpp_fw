#ifndef __CUDATIMEMEASURE_H_INCLUDED
#define __CUDATIMEMEASURE_H_INCLUDED

#include <cuda_runtime.h>

//////////////////////////////////////////////////
// CCudaTimeMeasure declaration

class CCudaTimeMeasure
{
public:
	CCudaTimeMeasure(cudaStream_t csStreamID = 0);
	~CCudaTimeMeasure();

public:
	float GetTimeout(bool bResetStart = false);

private:
	cudaEvent_t		m_ceStartEvent;
	cudaEvent_t		m_ceStopEvent;

	cudaStream_t	m_csStreamID;
};

#endif // __CUDATIMEMEASURE_H_INCLUDED
