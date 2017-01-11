#ifndef __H_DISPLAY__
#define __H_DISPLAY__

#define USER_INTERACTION_TIMEOUT 200

class display_t
{
public:
	static void turn_off( size_t nTimeOut = USER_INTERACTION_TIMEOUT );
	static void turn_on();

	static bool is_off() { return bDisplayIsOff; }

	static void change_state();

private:
	static bool save_bitmap(const char *szFileName, HBITMAP bmp, HPALETTE pal = NULL);

public:
	static void save_screen(const char *szFileName);

private:
	display_t();
	~display_t();

	static bool bDisplayIsOff;
};

#endif // __H_DISPLAY__
