#ifndef __MEMORY_BUFFER_H_INCLUDED
#define __MEMORY_BUFFER_H_INCLUDED

#include <stdlib.h>

//////////////////////////////////////////////////
// memory_buffer_t declaration

class memory_buffer_t
{
public:
	memory_buffer_t(size_t size) : m_buffer_size(size), m_buffer(NULL)
	{
		m_buffer = malloc(m_buffer_size);
	}

	~memory_buffer_t()
	{
		if (m_buffer)
		{
			free(m_buffer);
			m_buffer = NULL;
		}
	}

	operator void*() { return m_buffer; }

private:
	size_t m_buffer_size;
	void *m_buffer;
};

#endif // __MEMORY_BUFFER_H_INCLUDED
