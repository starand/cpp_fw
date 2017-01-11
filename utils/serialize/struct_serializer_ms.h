#ifndef __STRUCT_SERIALIZER_MS_H_INCLUDED
#define __STRUCT_SERIALIZER_MS_H_INCLUDED

#include <memory_stream.h>
#include <memory_buffer.h>
#include <string>
#include <xsocket.h>


//#define DEBUG


template <typename fieldT>
class field_helper_ms_t
{
public:
	static size_t get_size(const fieldT *value)
	{
            (void)value;
            return sizeof(fieldT);
	}

	static bool store(memory_stream_t &stream, const fieldT *value)
	{
		return stream.write((void *)value, get_size(value));
	}

	static bool load(memory_stream_t &stream, fieldT *value)
	{
		return stream.read(value, get_size(value));
	}
};

template <>
class field_helper_ms_t<std::string>
{
public:
	static size_t get_size(const std::string *value)
	{
		return sizeof(size_t) + value->length();
	}

	static bool store(memory_stream_t &stream, const std::string *value)
	{
		uint length = value->length();
		return stream.write(&length, sizeof(length)) && stream.write((void *)value->c_str(), value->length());
	}

	static bool load(memory_stream_t &stream, std::string *value)
	{
		uint length;
		if (!stream.read(&length, sizeof(length)))
		{
			return false;
		}

		value->resize(length);
		return stream.read((void *)value->c_str(), length);
	}
};


template <typename structT, typename fieldT, fieldT structT::*memberPtr>
class field_info_ms_t : public field_helper_ms_t<fieldT>
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

	static size_t get_size(const structT &structure)
	{
		return field_helper_ms_t<fieldT>::get_size(get_ptr(structure));
	}

	static bool store(memory_stream_t &stream, const structT &structure)
	{
		return field_helper_ms_t<fieldT>::store(stream, get_ptr(structure));
	}

	static bool load(memory_stream_t &stream, structT &structure)
	{
		return field_helper_ms_t<fieldT>::load(stream, get_ptr(structure));
	}
};

template <class structT>
class empty_info_ms_t
{
public:
        static size_t get_size(const structT &structure) { (void)structure; return 0; }

        static bool store(memory_stream_t &stream, const structT &structure) { (void)structure; (void)stream; return true; }
        static bool load(memory_stream_t &stream, structT &structure) { (void)structure; (void)stream; return true; }
};


template <typename structT,
	typename fieldT1 = empty_info_ms_t<structT>, typename fieldT2 = empty_info_ms_t<structT>, typename fieldT3 = empty_info_ms_t<structT>, typename fieldT4 = empty_info_ms_t<structT>,
	typename fieldT5 = empty_info_ms_t<structT>, typename fieldT6 = empty_info_ms_t<structT>, typename fieldT7 = empty_info_ms_t<structT>, typename fieldT8 = empty_info_ms_t<structT>,
	typename fieldT9 = empty_info_ms_t<structT>, typename fieldT10 = empty_info_ms_t<structT>
>
class serializer_ms_t
{
public:
	static bool store(memory_stream_t &stream, const structT &structure)
	{
		return fieldT1::store(stream, structure) && fieldT2::store(stream, structure) && fieldT3::store(stream, structure) && fieldT4::store(stream, structure)
			&& fieldT5::store(stream, structure) && fieldT6::store(stream, structure) && fieldT7::store(stream, structure) && fieldT8::store(stream, structure)
			&& fieldT9::store(stream, structure) && fieldT10::store(stream, structure);
	}

	static bool load(memory_stream_t &stream, structT &structure)
	{
		return fieldT1::load(stream, structure) && fieldT2::load(stream, structure) && fieldT3::load(stream, structure) && fieldT4::load(stream, structure)
			&& fieldT5::load(stream, structure) && fieldT6::load(stream, structure) && fieldT7::load(stream, structure) && fieldT8::load(stream, structure)
			&& fieldT9::load(stream, structure) && fieldT10::load(stream, structure);
	}

	static size_t get_size(const structT &structure)
	{
		return fieldT1::get_size(structure) + fieldT2::get_size(structure) + fieldT3::get_size(structure) + fieldT4::get_size(structure) + fieldT5::get_size(structure)
			+ fieldT6::get_size(structure) + fieldT7::get_size(structure) + fieldT8::get_size(structure) + fieldT9::get_size(structure) + fieldT10::get_size(structure);
	}

	static bool send(socket_t &socket, structT &structure)
	{
		uint size = get_size(structure);
		//LOG_TRACE("[protocol] Packet size: %u", size);

		if (socket.send(&size, sizeof(size)) != sizeof(size))
		{
			return false;
		}

		if (size == 0)
		{
			return true;
		}

		memory_buffer_t buffer(size);
		memory_stream_t stream(buffer, size);

		if (!store(stream, structure))
		{
			return false;
		}

		if (socket.send(buffer, size) != (long)size)
		{
			return false;
		}

		return true;
	}

	static bool recv(socket_t &socket, structT &structure)
	{
		uint size = 0;
		if (socket.recv(&size, sizeof(size)) != sizeof(size))
		{
			return false;
		}

#ifdef DEBUG
		cout << "Size = " << size;
#endif

		if (size == 0)
		{
			return true;
		}

		memory_buffer_t buffer(size);
	        if ( ((uint)socket.recv( (char*)(void*)buffer, size )) != size )
        	{
	            return false;
        	}

		memory_stream_t stream(buffer, size);
#ifdef DEBUG
		buffer.dump();
#endif
		if (!load(stream, structure))
		{
			return false;
		}

		return true;
	}
};


template<typename T>
struct struct_serializer_t
{
};


#define STRUCT_SERIALIZER_FIELD(field_type_, field_name_) \
	, field_info_ms_t<struct_type_t, field_type_, &struct_type_t::field_name_>

#define STRUCT_SERIALIZER_BEGIN(type_) \
	template<> struct struct_serializer_t<type_> \
	{ \
	typedef type_ struct_type_t; \
	typedef serializer_ms_t<type_

#define STRUCT_SERIALIZER_END() \
	> serializer_t; \
	};

#endif // __STRUCT_SERIALIZER_MS_H_INCLUDED
