#ifndef __LOGGER_H_INCLUDED
#define __LOGGER_H_INCLUDED

#include <types.h>
#include <stdio.h>


//#define STDOUT_LOGGER

#ifdef STDOUT_LOGGER
#define LOG_TRACE(msg, ...) printf("[TRACE] " msg "\n", ##__VA_ARGS__)
#define LOG_DEBUG(msg, ...) printf("[DBG] " msg "\n", ##__VA_ARGS__)
#define LOG_INFO(msg, ...) printf("[NFO] " msg "\n", ##__VA_ARGS__)
#define LOG_WARNING(msg, ...) printf("[WRN] " msg "\n", ##__VA_ARGS__)
#define LOG_ERROR(msg, ...) printf("[ERR] (%s:%u) " msg "\n", __FILE__, __LINE__, ##__VA_ARGS__)
#define LOG_FATAL(msg, ...) printf("[FATAL] (%s:%u) " msg "\n", __FILE__, __LINE__, ##__VA_ARGS__)
#else
#define LOG_TRACE(msg, ...) logger::log_message(logger::SEVERITY_TRACE, msg, ##__VA_ARGS__)
#define LOG_DEBUG(msg, ...) logger::log_message(logger::SEVERITY_DEBUG, msg, ##__VA_ARGS__)
#define LOG_INFO(msg, ...) logger::log_message(logger::SEVERITY_INFO, msg, ##__VA_ARGS__)
#define LOG_WARNING(msg, ...) logger::log_message(logger::SEVERITY_WARNING, msg, ##__VA_ARGS__)
#define LOG_ERROR(msg, ...) logger::log_message(logger::SEVERITY_ERR0R, msg, ##__VA_ARGS__)
#define LOG_FATAL(msg, ...) logger::log_message(logger::SEVERITY_FATAL, "(%s:%u) " msg, __FILE__, __LINE__, ##__VA_ARGS__)
#endif


#define LOGGER_PORT_DEF	10993


namespace logger
{
	enum ELOGSERVERITY
	{
		SEVERITY__MIN,

		SEVERITY_NONE = SEVERITY__MIN,
		SEVERITY_FATAL,
		SEVERITY_ERR0R,
		SEVERITY_WARNING,
		SEVERITY_INFO,
		SEVERITY_DEBUG,
		SEVERITY_TRACE,

		SEVERITY__MAX,
	};

	struct net_message_t
	{
		uint length;
		const char *message;

		net_message_t() : length(0), message(NULL) { }
	};


	bool initialize(const char *source = NULL, const char *config_file = NULL);
	void finalize();

	void set_loggers(const char *source, void *loggers);
	void *get_loggers();

	const char *get_source();
    void set_source( const std::string& name );

	void log_message(ELOGSERVERITY severity, const char *format, ...);
};


class logger_initializer_t
{
public:
	logger_initializer_t(const char *source = NULL, const char *config_file = NULL)
	{
		logger::initialize(source, config_file);
	}

	~logger_initializer_t()
	{
		logger::finalize();
	}
};

#define LOG_INIT(_SOURCE_, _CONFIGFILE_) logger_initializer_t __logger__(_SOURCE_, _CONFIGFILE_);

#endif // __LOGGER_H_INCLUDED
