#ifndef __REF_COUNTER_H_INCLUDED
#define __REF_COUNTER_H_INCLUDED


template <typename T>
class ref_counter_t
{
public:
	ref_counter_t(T *pointee)
	{
		ASSERT(pointee != NULL);
		m_storage = new ref_counter_storage_t();
		m_storage->m_count = 1;
		m_storage->m_pointee = pointee;
	}

	ref_counter_t(ref_counter_t<T> &ref_conuter)
	{
		m_storage = ref_conuter.m_storage;
		atomic_add(m_storage->m_count, 1);
	}

	~ref_counter_t()
	{
		ASSERT(m_storage != NULL); ASSERT(m_storage->m_pointee != NULL);
		if (atomic_add(m_storage->m_count, -1) == 1)
		{
			delete m_storage->m_pointee;
			delete m_storage;
		}
	}

	T *get()
	{
		ASSERT(m_storage != NULL); ASSERT(m_storage->m_pointee != NULL);
		return m_storage->m_pointee;
	}

private:
	ref_counter_t<T> &operator=(ref_counter_t<T> &ref_counter);

	typedef struct 
	{
		volatile long m_count;
		T *m_pointee;
	} 
	ref_counter_storage_t;

private:
	ref_counter_storage_t *m_storage;
};

#endif // __REF_COUNTER_H_INCLUDED
