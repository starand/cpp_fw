#ifndef __RASPBERRY_H_INCLUDED
#define __RASPBERRY_H_INCLUDED

#include "c_gpio.h"

enum GPIO
{
	GPIO_CE1 = 7,
	GPIO_MISO = 9,
	GPIO_MOSI = 10,
	GPIO_SCK = 11,
};


class gpio_t
{
public:
	gpio_t(int pin, int direction = OUTPUT, int pud = PUD_OFF);

public:
	void set(bool value);
	bool get();

private:
	int m_pin;
	int m_direction;
	int m_pud;
};

#endif // __RASPBERRY_H_INCLUDED
