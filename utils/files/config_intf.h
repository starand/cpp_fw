#ifndef __CONFIG_INTF_H_INCLUDED
#define __CONFIG_INTF_H_INCLUDED


class config_intf_t
{
public:
	virtual ~config_intf_t() { }

	virtual bool load_from_file(const string &file_name) = 0;
	virtual bool load_from_commandline(int argc, char *argv[]) = 0;

	virtual bool get_int_param(const string &param, int &value) = 0;
	virtual bool get_string_param(const string &param, string &value) = 0;
	virtual bool get_bool_param(const string &param, bool &value) = 0;
};

#endif // __CONFIG_INTF_H_INCLUDED
