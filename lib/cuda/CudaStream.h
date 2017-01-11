#ifndef __CUDASTREAM_H_INCLUDED
#define __CUDASTREAM_H_INCLUDED

//////////////////////////////////////////////////
// CCudaStream declaration

class CCudaStream
{
public:
	CCudaStream();
	~CCudaStream();

public:
	operator cudaStream_t() const { return m_csStream; }

	void Synchronize() const;

private:
	cudaStream_t	m_csStream;
};

#endif // __CUDASTREAM_H_INCLUDED
