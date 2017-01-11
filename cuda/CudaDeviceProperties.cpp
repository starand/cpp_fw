#include "StdAfx.h"

#include "CudaDeviceProperties.h"

//////////////////////////////////////////////////
// CCudaDeviceProperties implementation

CCudaDeviceProperties::CCudaDeviceProperties(int nDeviceIndex):
	m_nDeviceIndex(nDeviceIndex)
{
	Init();
}

CCudaDeviceProperties::~CCudaDeviceProperties()
{

}


void CCudaDeviceProperties::Init()
{
	const int nDeviceIndex = GetDeviceIndex();

	int nDevicesCount;
	cudaCheckError(cudaGetDeviceCount(&nDevicesCount));

	if (nDeviceIndex < nDevicesCount)
	{
		cudaCheckError(cudaGetDeviceProperties(&m_dpProperties, nDeviceIndex));
	}
	else
	{
		cout << "Incorrect device index " << nDeviceIndex << ". Proper index in range 0 .. " << nDevicesCount << endl;
	}
}
