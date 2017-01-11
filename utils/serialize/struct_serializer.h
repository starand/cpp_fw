#ifndef __STRUCT_SERIALIZER_H_INCLUDED
#define __STRUCT_SERIALIZER_H_INCLUDED

#include "xsocket.h"


template <typename fieldT> 
class field_helper_t
{
public:
	static bool send_field(socket_t &socket, const fieldT *value)
	{
		return socket.Send((const char *)value, sizeof(fieldT));
	}

	static bool recv_field(socket_t &socket, fieldT *value)
	{
		return socket.Recv((char *)value, sizeof(fieldT));
	}

	static size_t get_size(const fieldT *value)
	{
		return sizeof(fieldT);
	}
};

template <> 
class field_helper_t<string>
{
public:
	static bool send_field(socket_t &socket, const string *value)
	{
		uint length = value->length();
		return socket.send((void *)&length, sizeof(length)) == sizeof(length)
			&& socket.send((void *)value->c_str(), length) == length;
	}

	static bool recv_field(socket_t &socket, string *value)
	{
		uint length;
		if (socket.recv((char *)&length, sizeof(uint)) != sizeof(uint)) return false;
		value->reserve(length);
		return socket.recv((void *)value->c_str(), length) == length;
	}

	static size_t get_size(const string *value)
	{
		return sizeof(size_t) + value->length();
	}
};


template <typename structT, typename fieldT, fieldT structT::*memberPtr> 
class field_info_t : public field_helper_t<fieldT>
{
public:
	static fieldT *get_ptr(structT &structure)
	{
		return &(structure.*memberPtr);
	}

	static const fieldT *get_ptr(const	structT &structure)
	{
		return &(structure.*memberPtr);
	}

	static bool send_field(socket_t &socket, const structT &structure)
	{
		return field_helper_t<fieldT>::send_field(socket, get_ptr(structure));
	}

	static bool recv_field(socket_t &socket, structT &structure)
	{
		return field_helper_t<fieldT>::recv_field(socket, get_ptr(structure));
	}

	static size_t get_size(const structT &structure)
	{
		return field_helper_t<fieldT>::get_size(get_ptr(structure));
	}
};

template <class structT> 
class empty_info_t
{
public:
	static bool send_field(socket_t &socket, const structT &structure) { return true; }
	static bool recv_field(socket_t &socket, structT &structure) { return true; }
	static size_t get_size(const structT &structure) { return 0; }
};


template <typename structT, 
	typename fieldT1 = empty_info_t<structT>, typename fieldT2 = empty_info_t<structT>, typename fieldT3 = empty_info_t<structT>, typename fieldT4 = empty_info_t<structT>, 
	typename fieldT5 = empty_info_t<structT>, typename fieldT6 = empty_info_t<structT>, typename fieldT7 = empty_info_t<structT>, typename fieldT8 = empty_info_t<structT>, 
	typename fieldT9 = empty_info_t<structT>, typename fieldT10 = empty_info_t<structT> 
>
class serializer_t
{
public:
	static bool send_packet(socket_t &socket, const structT &structure)
	{
		return fieldT1::send_field(socket, structure) && fieldT2::send_field(socket, structure) && fieldT3::send_field(socket, structure) && fieldT4::send_field(socket, structure)
			&& fieldT5::send_field(socket, structure) && fieldT6::send_field(socket, structure) && fieldT7::send_field(socket, structure) && fieldT8::send_field(socket, structure)
			&& fieldT9::send_field(socket, structure) && fieldT10::send_field(socket, structure);
	}

	static bool recv_packet(socket_t &socket, structT &structure)
	{
		return fieldT1::recv_field(socket, structure) && fieldT2::recv_field(socket, structure) && fieldT3::recv_field(socket, structure) && fieldT4::recv_field(socket, structure)
			&& fieldT5::recv_field(socket, structure) && fieldT6::recv_field(socket, structure) && fieldT7::recv_field(socket, structure) && fieldT8::recv_field(socket, structure)
			&& fieldT9::recv_field(socket, structure) && fieldT10::recv_field(socket, structure);
	}

	static size_t get_size(const structT &structure)
	{
		size_t size = fieldT1::get_size(structure) + fieldT2::get_size(structure) + fieldT3::get_size(structure) + fieldT4::get_size(structure) + fieldT5::get_size(structure)
			+ fieldT6::get_size(structure) + fieldT7::get_size(structure) + fieldT8::get_size(structure) + fieldT9::get_size(structure) + fieldT10::get_size(structure);
		return size;
	}
};


#define STRUCT_SERIALIZER_BEGIN(type_) \
	typedef type_ struct_type_t; \
	typedef serializer_t<type_

#define STRUCT_SERIALIZER_FIELD(field_type_, field_name_) \
	, field_info_t<struct_type_t, field_type_, &struct_type_t::field_name_>

#define STRUCT_SERIALIZER_END(result_type_) \
	> result_type_;

#endif // __STRUCT_SERIALIZER_H_INCLUDED
