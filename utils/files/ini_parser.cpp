#include "StdAfx.h"
#include "ini_parser.h"

#include <logger.h>
#include <asserts.h>
#include <fileutils.h>
#include <strutils.h>
#include <fstream>
#include <stdlib.h>

ini_parser_t::ini_parser_t()
	: m_storage()
{

}

ini_parser_t::~ini_parser_t()
{
	clear();
}


bool ini_parser_t::load(const string& ini_file)
{
	ifstream fin(ini_file.c_str());
	if (!fin.is_open())
	{
		string folder;
		if (!FileUtils::GetBinaryDir(folder))
		{
			LOG_ERROR("Could not retrieve path to binary");
			return false;
		}

		string full_name = folder + ini_file;
		fin.open(full_name.c_str());
		if (!fin.is_open())
		{
			LOG_ERROR("Could not open fiile %s", ini_file.c_str());
			return false;
		}
	}

	string row, key, value;
	section_storage_t *section = NULL;

	while (!fin.eof() && fin.good())
	{
		getline(fin, row);
		if (row.empty() || row[0] == ';')
		{
			continue;
		}

		if (row[0] == '[')
		{
			StrUtils::RemoveBrackets(row, '[', ']');
			ini_storage_t::const_iterator itStorage = m_storage.find(row);
			if (itStorage == m_storage.end())
			{
				section = new section_storage_t();
				m_storage.insert(ini_storage_t::value_type(row, section));
			}
			else
			{
				section = itStorage->second;
			}
			continue;
		}

		if (!section)
		{
			LOG_ERROR("Could not find section before key-value");
			return false;
		}

		StrUtils::ParseNameValuePair(row.c_str(), key, value);
		if (!key.empty())
		{
			(*section)[key] = value;
		}
	}

	return true;
}


section_storage_t *ini_parser_t::get_section(const char *name) const
{
	ASSERT(name != NULL);
	ini_storage_t::const_iterator itStorage = m_storage.find(name);
	return itStorage == m_storage.end() ? NULL : itStorage->second;
}


bool ini_parser_t::get_value(const char *section_name, const char *key, string& value) const
{
	ini_storage_t::const_iterator itStorage = m_storage.find(section_name);
	if (itStorage == m_storage.end())
	{
		return false;
	}

	ASSERT(itStorage->second != NULL);
	section_storage_t& section = *(itStorage->second);

	section_storage_t::const_iterator itSection = section.find(key);
	if (itSection == section.end())
	{
		return false;
	}

	value = itSection->second;
	return true;
}

bool ini_parser_t::get_value(const char *section_name, const char *key, int& value) const
{
	string str_value;
	if (!get_value(section_name, key, str_value))
	{
		return false;
	}

	value = atoi(str_value.c_str());
	return true;
}


void ini_parser_t::clear()
{
	for (ini_storage_t::iterator itStorage = m_storage.begin(); itStorage != m_storage.end(); ++itStorage)
	{
		delete itStorage->second;
	}

	m_storage.clear();
}
