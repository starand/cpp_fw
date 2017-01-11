#include "gpio.h"
#include <iostream>

//#define GPIO_DEBUG


static bool g_bGpioInitialize = false;

class gpio_initializer_t
{
public:
	gpio_initializer_t()
	{
		int result = setup();
		switch (result)
		{
		case SETUP_OK:
			g_bGpioInitialize = true;
#ifdef GPIO_DEBUG
			std::cout << "[ + ]  GPIO initialized" << std::endl;
#endif
			break;
		default:
			std::cout << "Unable to initialize GPIO. Error : " << result << std::endl;
		}
	}

	~gpio_initializer_t()
	{
		cleanup();
#ifdef GPIO_DEBUG
		std::cout << "[ + ]  GPIO cleaned up" << std::endl;
#endif
	}
} g_gpio_initializer;


gpio_t::gpio_t(int pin, int direction /*= OUTPUT*/, int pud /*= PUD_OFF*/)
	: m_pin(pin)
	, m_direction(direction)
	, m_pud(pud)
{
	setup_gpio(m_pin, m_direction, m_pud);
}


void gpio_t::set(bool value)
{
	output_gpio(m_pin, value ? HIGH : LOW);
}

bool gpio_t::get()
{
	return input_gpio(m_pin) != LOW;
}
