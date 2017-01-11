#include "StdAfx.h"

#include "logger.h"
#include <asserts.h>

#include <threading.h>
#include <lock_queue.h>
#include <ref_counter.h>
#include <scoped_ptr.h>
#include <net/xsocket.h>
#include <ini_parser.h>
#include <utils.h>
#include <stdlib.h>
#include <iostream>
#include <cstring>

#ifdef WINDOWS
#	include <Winbase.h>
#	pragma warning(disable : 4996)
#else
#	include <sys/time.h>
#	include <stdarg.h>
#endif


#define SECTION_CONSOLE	"console"
#define SECTION_FILE	"file"
#define SECTION_NET		"net"

#define KEY_SEVERITY	"severity"
#define KEY_LOG_FILE	"log_file"
#define KEY_SERVER		"server"
#define KEY_PORT		"port"


namespace logger {

const char *g_aszSeverityDescription[SEVERITY__MAX] =
{
	"",			// LS_NONE = LS__MIN,

	"[FATAL]",	// LS_FATAL,
	"[ERR]",	// LS_ERROR,
	"[WRN]",	// LS_WARNING,
	"[NFO]",	// LS_INFO,
	"[DBG]",	// LS_DEBUG,
	"[TRACE]",  // LS_TRACE
};

const uint g_auiSeverityDescLength[SEVERITY__MAX] =
{
	0, // LS__MIN

	(uint)strlen(g_aszSeverityDescription[SEVERITY_FATAL]),
	(uint)strlen(g_aszSeverityDescription[SEVERITY_ERR0R]),
	(uint)strlen(g_aszSeverityDescription[SEVERITY_WARNING]),
	(uint)strlen(g_aszSeverityDescription[SEVERITY_INFO]),
	(uint)strlen(g_aszSeverityDescription[SEVERITY_DEBUG]),
	(uint)strlen(g_aszSeverityDescription[SEVERITY_TRACE]),
};


//////////////////////////////////////////////////
// log_time_t implementation

struct log_time_t
{
	log_time_t()
	{
		get_system_time();
	}

	void get_system_time()
	{
#ifdef WINDOWS
		SYSTEMTIME win_tm;
		GetLocalTime(&win_tm);
		year = win_tm.wYear;
		month = win_tm.wMonth;
		day = win_tm.wDay;
		hour = win_tm.wHour;
		minute = win_tm.wMinute;
		second = win_tm.wSecond;
		millisecond = win_tm.wMilliseconds;
#else
		timeval tv;
		gettimeofday(&tv, NULL);

		time_t tt = tv.tv_sec;
		suseconds_t us = tv.tv_usec;

		tm lin_tm;
		localtime_r(&tt, &lin_tm);

		year = lin_tm.tm_year+1900;
		month = lin_tm.tm_mon+1;
		day = lin_tm.tm_mday;
		hour = lin_tm.tm_hour;
		minute = lin_tm.tm_min;
		second = lin_tm.tm_sec;
		millisecond = us;
#endif
	}

	int year;
	int month;
	int day;

	int hour;
	int minute;
	int second;
	int millisecond;
};


//////////////////////////////////////////////////
// msg_foramter_t implementation

class msg_foramter_t
{
public:
	static const char *format_message(ELOGSERVERITY severity, const char *source, const char *message, uint thread_id)
	{
		if (message == NULL)
		{
			return NULL;
		}

		size_t message_length = 24/*timestamp*/ + g_auiSeverityDescLength[severity] + (source ? strlen(source) : 0) + strlen(message) + 1 + 10/*thread_id*/ + 6;
		char *buffer = (char*)malloc(message_length);

		struct log_time_t dt;
		sprintf(buffer, "%04d-%02d-%02d %02d:%02d:%02d %s {%s} %u %s", dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second, g_aszSeverityDescription[severity], (source ? source : ""), thread_id, message);
		return buffer;
	}

};


//////////////////////////////////////////////////
// log_item_t implementation

class log_item_t
{
public:
	log_item_t(const char *message)
		: m_message(message)
	{
	}

	~log_item_t()
	{
		delete[] m_message;
	}

public:
	const char *get() const { return m_message; }

private:
	log_item_t(const log_item_t& );
	log_item_t& operator=(const log_item_t& );

private:
	const char *m_message;
};

typedef ref_counter_t<log_item_t> log_item_ref_t;


//////////////////////////////////////////////////
// logger_base_t implementation

log_item_ref_t *stop_msg = new log_item_ref_t(new log_item_t(""));

class logger_base_t : public thread_base_t
{
public:
	logger_base_t() : m_severity(SEVERITY_DEBUG), m_storage_opened(false) { }
	virtual ~logger_base_t() { }

public:
	virtual void do_run()
	{
		while (!is_stopping())
		{
			do
			{
				scoped_ptr_t<log_item_ref_t> log_item(m_queue.pop());
				const char *message = log_item->get()->get();

				if (message[0] == 0)
				{
					break; // stopping ?
				}

				store_message(message);
			}
			while (!m_queue.empty());

			flush();
		}

	}

	virtual void do_stop()
	{
		log_item_ref_t *log_item_ref = new log_item_ref_t(*stop_msg);
		m_queue.push(log_item_ref);
	}

	void add_message(log_item_ref_t *log_item)
	{
		m_queue.push(log_item);
	}

public:
	bool open_storage()
	{
		m_storage_opened = do_open_storage();
		return m_storage_opened;
	}

	void close_storage()
	{
		m_storage_opened = false;
		do_close_storage();
	}

	bool is_storage_opened() const
	{
		return m_storage_opened;
	}

protected:
	virtual bool do_open_storage() = 0;
	virtual void do_close_storage() = 0;

	virtual bool store_message(const char *message) = 0;

	virtual void flush() { }

public:
	ELOGSERVERITY get_severity() const { return m_severity; }
	void set_severity(ELOGSERVERITY severity)
	{
		if (severity >= SEVERITY__MAX) severity = SEVERITY_ERR0R;
		m_severity = severity;
	}

private:
	lock_queue_t<log_item_ref_t *> m_queue;
	ELOGSERVERITY m_severity;
	bool m_storage_opened;
};


//////////////////////////////////////////////////
// console_logger_t implementation

class console_logger_t : public logger_base_t
{
private:
	virtual bool do_open_storage()
	{
		return true;
	}

	virtual void do_close_storage() { }

	virtual bool store_message(const char *message)
	{
		ASSERT(message != NULL);
		cout << message << endl;
		cout.flush();
		return true;
	}
};

//////////////////////////////////////////////////
// net_logger_t implementation

class net_logger_t : public logger_base_t
{
public:
	net_logger_t(const string& server, ushort port)
		: m_server(server)
		, m_port(port)
	{
		ASSERT(!server.empty()); ASSERT(port > 0);
	}

private:
	virtual bool do_open_storage()
	{
		ASSERT(m_socket.is_connected() == false);
		if (!m_socket.connect(m_server, m_port))
		{
			cerr << "Could not connect to logger server " << endl;
			return false;
		}

		m_local_endpoint = " -- " + m_socket.get_local_endpoint();
		return true;
	}

	virtual void do_close_storage()
	{
		m_socket.close();
	}

	virtual bool store_message(const char *message)
	{
		if (message == NULL || !m_socket.is_connected())
		{
			return false;
		}

		uint message_length = strlen(message);
		uint full_length = message_length + m_local_endpoint.length();

		if (m_socket.send(&full_length, sizeof(full_length)) != sizeof(full_length))
		{
			return false;
		}

		if (m_socket.send((void*)message, message_length) != (long)message_length)
		{
			return false;
		}

		if (m_socket.send((void*)m_local_endpoint.c_str(), m_local_endpoint.length()) != (long)m_local_endpoint.length())
		{
			return false;
		}

		return true;
	}

private:
	const string m_server;
	const ushort m_port;

	socket_t m_socket;
	string m_local_endpoint;
};

//////////////////////////////////////////////////
// file_logger_t implementation

class file_logger_t : public logger_base_t
{
public:
	file_logger_t(const string& file_name)
		: m_file_name(file_name)
		, m_file(NULL)
	{
		ASSERT(!m_file_name.empty());

		if (m_file_name == "~/self.log")
		{
			string binary;
			utils::get_binary_name(binary);
			m_file_name = binary.c_str() + string(".log");
		}
	}

private:
	virtual bool do_open_storage()
	{
		ASSERT(m_file == NULL);
		m_file = fopen(m_file_name.c_str(), "ab+");
		if (m_file == NULL)
		{
			cerr << "Could not open log file" << endl;
			return false;
		}

		return true;
	}

	virtual void do_close_storage()
	{
		if (m_file != NULL)
		{
			fclose(m_file);
			m_file = NULL;
		}
	}

	virtual bool store_message(const char *message)
	{
		if (message == NULL || m_file == NULL)
		{
			return false;
		}

		fprintf(m_file, "%s\r\n", message);
		return true;
	}

	virtual void flush()
	{
		if (m_file != NULL)
		{
			fflush(m_file);
			utils::sleep_ms(100);
		}
	}

private:
	string m_file_name;
	FILE *m_file;
};


//////////////////////////////////////////////////
// logger interface implementation

enum ELOGGERTYPE
{
	LOGGERTYPE_CONSOLE,
	LOGGERTYPE_FILE,
	LOGGERTYPE_NET,

	LOGGER_TYPE__MAX
};

bool g_bLoggerInitialized = false;
string g_sSource;

logger_base_t *g_aLoggers[LOGGER_TYPE__MAX] = { 0 };
logger_base_t **g_pLoggers = g_aLoggers;

void set_loggers(const char *source, void *loggers)
{
	ASSERT(loggers != NULL); ASSERT(source != NULL);
	g_sSource = source;

	g_pLoggers = (logger_base_t **)loggers;
	g_bLoggerInitialized = true;
}

void *get_loggers()
{
	return g_pLoggers;
}

const char *get_source()
{
	return g_sSource.c_str();
}

void set_source( const std::string& name )
{
    g_sSource = name;
}


#define INIT_LOGGER(_LOG_TYPE_, _INITIATOR_, _SEVERITY_) \
	ASSERT(g_aLoggers[_LOG_TYPE_] == NULL); \
	g_aLoggers[_LOG_TYPE_] = new _INITIATOR_; \
	if (g_aLoggers[_LOG_TYPE_]->open_storage()) {\
		g_aLoggers[_LOG_TYPE_]->set_severity(_SEVERITY_); \
		g_aLoggers[_LOG_TYPE_]->start(); \
	}


bool initialize(const char *source/* = NULL*/, const char *config_file/* = NULL*/)
{
	if (g_bLoggerInitialized)
	{
		return false;
	}

	if (source)
	{
		g_sSource = source;
	}

	if (config_file)
	{
		ini_parser_t config;
		config.load(config_file);

		int severity;
		if (config.get_value(SECTION_CONSOLE, KEY_SEVERITY, severity))
		{
			INIT_LOGGER(LOGGERTYPE_CONSOLE, console_logger_t(), (ELOGSERVERITY)severity);
			g_bLoggerInitialized = true;
		}

		string log_file;
		if (config.get_value(SECTION_FILE, KEY_LOG_FILE, log_file))
		{
			config.get_value_def(SECTION_FILE, KEY_SEVERITY, severity, (int)SEVERITY_ERR0R);
			INIT_LOGGER(LOGGERTYPE_FILE, file_logger_t(log_file), (ELOGSERVERITY)severity);
			g_bLoggerInitialized = true;
		}

		int port;
		string server;
		if (config.get_value(SECTION_NET, KEY_SERVER, server))
		{
			config.get_value_def(SECTION_NET, KEY_SEVERITY, severity, (int)SEVERITY_ERR0R);
			config.get_value_def(SECTION_NET, KEY_PORT, port, LOGGER_PORT_DEF);
			INIT_LOGGER(LOGGERTYPE_NET, net_logger_t(server, port), (ELOGSERVERITY)severity);
			g_bLoggerInitialized = true;
		}
	}

	if (!g_bLoggerInitialized)
	{
		INIT_LOGGER(LOGGERTYPE_CONSOLE, console_logger_t(), SEVERITY_DEBUG);
		g_bLoggerInitialized = true;
	}

	{
		string binary;
		utils::get_binary_name(binary);
		LOG_INFO("=== Starting %s. PID: %u", binary.c_str(), utils::get_current_process_id());
	}

	return g_bLoggerInitialized;
}

void finalize()
{
	{
		string binary;
		utils::get_binary_name(binary);
		LOG_INFO("=== Stopping %s", binary.c_str());
	}

	for (uint idx = 0; idx < LOGGER_TYPE__MAX; ++idx)
	{
		if (g_aLoggers[idx])
		{
			logger_base_t *logger = g_aLoggers[idx];
			if (logger->is_storage_opened())
			{
				g_aLoggers[idx] = NULL;
				logger->stop();
				logger->wait();
				logger->close_storage();
			}

			FREE_POINTER(logger);
		}
	}
}


void log_message(ELOGSERVERITY severity, const char *format, ...)
{
	const size_t LOGGER_BUFFER_SIZE = 4096;
	char buffer[LOGGER_BUFFER_SIZE];

	va_list ptr;
	va_start(ptr, format);
	vsnprintf(buffer, LOGGER_BUFFER_SIZE, format, ptr);
	va_end(ptr);
	buffer[LOGGER_BUFFER_SIZE - 1] = 0;

	const char *message = msg_foramter_t::format_message(severity, g_sSource.c_str(), buffer, get_current_thread_id());
	log_item_ref_t item_ref(new log_item_t(message));

	for (uint idx = 0; idx < LOGGER_TYPE__MAX; ++idx)
	{
		if (g_pLoggers[idx] && g_pLoggers[idx]->is_storage_opened())
		{
			if (g_pLoggers[idx]->get_severity() >= severity)
			{
				g_pLoggers[idx]->add_message(new log_item_ref_t(item_ref));
			}
		}
	}
}


class logger_funalizer_t
{
public:
	~logger_funalizer_t()
	{
		logger::finalize();
	}
};

logger_funalizer_t finalizer;

};
