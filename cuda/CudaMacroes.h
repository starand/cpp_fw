#ifndef __CUDAMACROES_H_INCLUDED
#define __CUDAMACROES_H_INCLUDED

#include <iostream>

using namespace std;

#define cudaErrorCheck cudaCheckError

template<typename TCudaError>
void cudaCheckError(TCudaError ceError)
{
	static int nCallIndex = 0;
	++nCallIndex;

	if (cudaSuccess != ceError)
	{
		std::cout << "[" << nCallIndex << "] CUDA runtime failed with code : " << ceError << " - " << cudaGetErrorString(ceError) << std::endl;
		exit(-1);
	}
}

template<typename TCudaError>
void CHEK_CUDA_ERROR()
{
	TCudaError ceLastError = cudaGetLastError();
	cudaCheckError(ceLastError);
}

#endif // __CUDAMACROES_H_INCLUDED
