#include "StdAfx.h"
#include "CudaStream.h"

//////////////////////////////////////////////////
// CCudaStream implementation

CCudaStream::CCudaStream()
{
	cudaErrorCheck(cudaStreamCreate(&m_csStream));
}

CCudaStream::~CCudaStream()
{
	cudaErrorCheck(cudaStreamDestroy(m_csStream));
}


void CCudaStream::Synchronize() const
{
	cudaErrorCheck(cudaStreamSynchronize(m_csStream));
}
