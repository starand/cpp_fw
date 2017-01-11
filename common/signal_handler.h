#ifndef __SIGNAL_HANDLER_H_INCLUDED
#define __SIGNAL_HANDLER_H_INCLUDED

#include <threading.h>

event_t g_stop_event;

#ifdef WINDOWS
BOOL WINAPI default_console_handler(DWORD ctrl_type)
{
	switch (ctrl_type)
	{
	case CTRL_CLOSE_EVENT:
	case CTRL_BREAK_EVENT:
	case CTRL_C_EVENT:
		cout << "Ctrl + C .." << endl;
		g_stop_event.set();
		return TRUE;
		break;
	}

	return FALSE;
}
#else
#	include <signal.h>
void default_signal_handler(int sig)
{
	cout << "Ctrl + C .." << endl;
	g_stop_event.set();
}
#endif

class signal_handler_t
{
#ifdef WINDOWS
	friend BOOL WINAPI default_console_handler(DWORD ctrl_type);
#endif
public:
	static bool install()
	{
		if (!m_installed)
		{
#ifdef WINDOWS
			if (!::SetConsoleCtrlHandler(default_console_handler, TRUE))
#else
			struct sigaction sig_act;
			memset(&sig_act, 0, sizeof(sig_act));
			sig_act.sa_handler = default_signal_handler;

			if (sigaction(SIGTERM,& sig_act, NULL) != 0 || sigaction(SIGINT,& sig_act, NULL) != 0)
#endif
			{
				return false;
			}

		}

		m_installed = true;
		return true;
	}

	static void uninstall()
	{
#ifdef WINDOWS
		::SetConsoleCtrlHandler(NULL, FALSE);
#else
		struct sigaction sig_act;
		memset(&sig_act, 0, sizeof(sig_act));
		sig_act.sa_handler = SIG_DFL; // default

		sigaction(SIGTERM,& sig_act, NULL);
		sigaction(SIGINT,& sig_act, NULL);
#endif
		m_installed = false;
	}

	void wait_interrupt_signal()
	{
		g_stop_event.wait();
	}

private:
	static bool m_installed;
};

bool signal_handler_t::m_installed = false;





#endif // __SIGNAL_HANDLER_H_INCLUDED

