#ifndef __LIBRARY_H_INCLUDED
#define __LIBRARY_H_INCLUDED


//////////////////////////////////////////////////
// library_t declaration

class library_t
{
public:
	library_t();
	~library_t();

public:
	bool load(const char *name);
	bool unload();

	bool is_loaded();

	void *load_function(const char *name);

private:
	void *m_handle;
};

#endif // __LIBRARY_H_INCLUDED
