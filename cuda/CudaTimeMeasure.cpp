#include "StdAfx.h"
#include "CudaTimeMeasure.h"

#include "CudaMacroes.h"

//////////////////////////////////////////////////
// CCudaTimeMeasure implementation

CCudaTimeMeasure::CCudaTimeMeasure(cudaStream_t csStreamID/* = 0*/):
	m_ceStartEvent(NULL),
	m_ceStopEvent(NULL),
	m_csStreamID(csStreamID)
{
	cudaCheckError(cudaEventCreate(&m_ceStartEvent));
	cudaCheckError(cudaEventCreate(&m_ceStopEvent));

	cudaCheckError(cudaEventRecord(m_ceStartEvent, m_csStreamID));
}

CCudaTimeMeasure::~CCudaTimeMeasure()
{
	cudaCheckError(cudaEventDestroy(m_ceStartEvent));
	cudaCheckError(cudaEventDestroy(m_ceStopEvent));
}


float CCudaTimeMeasure::GetTimeout(bool bResetStart/* = false*/)
{
	cudaCheckError(cudaEventRecord(m_ceStopEvent, m_csStreamID));
	cudaCheckError(cudaEventSynchronize(m_ceStopEvent));

	float fElapsedTime = 0.0f;
	cudaCheckError(cudaEventElapsedTime(&fElapsedTime, m_ceStartEvent, m_ceStopEvent));

	if (bResetStart)
	{
		cudaCheckError(cudaEventRecord(m_ceStartEvent, m_csStreamID));
	}

	return fElapsedTime;
}
