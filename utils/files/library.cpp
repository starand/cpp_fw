#include "StdAfx.h"
#include "library.h"
#include "asserts.h"

#ifdef WINDOWS
#	include <Windows.h>
#else
#	include <dlfcn.h>
#endif

//////////////////////////////////////////////////
// library_t implementation

library_t::library_t()
	: m_handle(NULL)
{

}

library_t::~library_t()
{
}


bool library_t::load(const char *name)
{
	ASSERT(m_handle == NULL && name != NULL);

#ifdef WINDOWS
	m_handle = ::LoadLibraryA(name);
#else
	m_handle = dlopen(name, RTLD_LAZY);
#endif

	return m_handle != NULL;
}

bool library_t::unload()
{
	ASSERT(m_handle != NULL);

#ifdef WINDOWS
	bool result = ::FreeLibrary((HMODULE)m_handle) != 0;
#else
	bool result = dlclose(m_handle) == 0;
#endif

	m_handle = NULL;
	return result;
}


bool library_t::is_loaded()
{
	return m_handle != NULL;
}


void *library_t::load_function(const char *name)
{
	ASSERT(m_handle != NULL && name != NULL);

#ifdef WINDOWS
    return (void*)::GetProcAddress((HMODULE)m_handle, name);
#else
	return dlsym(m_handle, name);
#endif
}
