#ifndef __CUDADEVICEPROPERTIES_H_INCLUDED
#define __CUDADEVICEPROPERTIES_H_INCLUDED

//////////////////////////////////////////////////
// CCudaDeviceProperties declaration

class CCudaDeviceProperties
{
public:
	CCudaDeviceProperties(int nDeviceIndex = 0);
	~CCudaDeviceProperties();

private:
	void Init();

public:
// device
	string GetDeviceName() const { return m_dpProperties.name; }

	int GetMajorVer() const { return m_dpProperties.major; }
	int GetMinorVer() const { return m_dpProperties.minor; }

	bool GetDeviceOverlap() const { return m_dpProperties.deviceOverlap != 0; }
	bool GetIsIntegrated() const { return m_dpProperties.integrated != 0; }

	bool CanMapHostMemory() const { return m_dpProperties.canMapHostMemory != 0; }

	bool CanRunKernelsConcurrent() const { return m_dpProperties.concurrentKernels != 0; }
	bool KernelExecTimeoutEnabled() const { return m_dpProperties.kernelExecTimeoutEnabled != 0; }

	size_t GetMultiprocessorsCount() const { return (size_t)m_dpProperties.multiProcessorCount; }

// memory
	size_t GetGlobalMemorySize() const { return m_dpProperties.totalGlobalMem; }
	size_t GetConstMemorySize() const { return m_dpProperties.totalConstMem; }

// block methods
	size_t GetMaxSharedMemorySize() const { return m_dpProperties.sharedMemPerBlock; }
	size_t GetRegistersPerBlock() const { return (size_t)m_dpProperties.regsPerBlock; }
	size_t GetMaxThreadsCount() const { return m_dpProperties.maxThreadsPerBlock; }

	size_t GetMaxBlockSizeDimX() const { return m_dpProperties.maxThreadsDim[0]; }
	size_t GetMaxBlockSizeDimY() const { return m_dpProperties.maxThreadsDim[1]; }
	size_t GetMaxBlockSizeDimZ() const { return m_dpProperties.maxThreadsDim[2]; }

// grid methods
	size_t GetMaxGridSizeDimX() const { return m_dpProperties.maxGridSize[0]; }
	size_t GetMaxGridSizeDimY() const { return m_dpProperties.maxGridSize[1]; }
	size_t GetMaxGridSizeDimZ() const { return m_dpProperties.maxGridSize[2]; }

// warp
	size_t GetWarpSize() const { return (size_t)m_dpProperties.warpSize; }

// other
	size_t GetMaxPitchSize() const { return m_dpProperties.memPitch; }



private:
	int GetDeviceIndex() const { return m_nDeviceIndex; }

private:
	int				m_nDeviceIndex;
	cudaDeviceProp	m_dpProperties;
};

#endif // __CUDADEVICEPROPERTIES_H_INCLUDED
