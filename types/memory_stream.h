#ifndef __MEMORY_STREAM_H_INCLUDED
#define __MEMORY_STREAM_H_INCLUDED

#include <string.h>


class memory_stream_t
{
public:
	memory_stream_t(void *buffer, size_t buffer_size)
		: m_buffer((char*)buffer)
		, m_buffer_size(buffer_size)
		, m_position(0)
	{
	}
	
	bool read(void *out_buffer, size_t count)
	{
		size_t bytes_left = m_buffer_size - m_position;
		size_t bytes_read = bytes_left < count ? bytes_left : count;
		memcpy(out_buffer, m_buffer + m_position, bytes_read);
		m_position += bytes_read;
		return bytes_read == count;
	}

	bool write(void *in_buffer, size_t count)
	{
		size_t bytes_left = m_buffer_size - m_position;
		size_t bytes_write = bytes_left < count ? bytes_left : count;
		memcpy(m_buffer + m_position, in_buffer, bytes_write);
		m_position += bytes_write;
		return bytes_write == count;
	}

	void reset_position() { m_position = 0; }

	void dump()
	{
		for (size_t idx = 0; idx < m_buffer_size; ++idx)
		{
            //cout << m_buffer[idx] << ' ';
		}
	}

private:
	char	*m_buffer;
	size_t	m_buffer_size;
	size_t	m_position;
};

#endif // __MEMORY_STREAM_H_INCLUDED
