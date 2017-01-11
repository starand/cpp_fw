#ifndef __INI_PARSER_H_INCLUDED
#define __INI_PARSER_H_INCLUDED

#include <map>
#include <string_ic.h>


typedef std::map<string_ic, std::string> section_storage_t;
typedef std::map<string_ic, section_storage_t*> ini_storage_t;

class ini_parser_t
{
public:
	ini_parser_t();
	~ini_parser_t();

public:
	bool load(const string& ini_file);

	section_storage_t *get_section(const char *name) const;

	bool get_value(const char *section_name, const char *key, string& value) const;
	bool get_value(const char *section_name, const char *key, int& value) const;

	template <typename T>
	void get_value_def(const char *section_name, const char *key, T& value, const T& def_value) const
	{
		if (!get_value(section_name, key, value))
		{
			value = def_value;
		}
	}

private:
	void clear();

private:
	ini_storage_t m_storage;
};

#endif // __INI_PARSER_H_INCLUDED
